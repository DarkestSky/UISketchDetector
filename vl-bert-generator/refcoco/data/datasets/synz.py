import json
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer
import logging

def get_category_dict(category, i):
    return {"id": i, "supercategory": "none", "name": category}


class SynZ(Dataset):
    def __init__(self, sketch_root_path, prediction_path, description_path, synz_annotations_path,
                 pretrained_model_name=None, transform = None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, test_mode=False, **kwargs) -> None:
        '''
            基于 SynZ 生成的草图及检测结果的数据集, 引入了 Screen2Words 中的文本描述数据
        
            sketch_path: SynZ 生成的草图的路径, 将使用 Faster R-CNN 提取特征
            prediction_path: SynZ 对草图的检测结果的路径, 使用 pth 文件
            description_path: Screen2Words 数据的路径
            synz_annotations_path: SynZ 数据集中的 annotations json 文件路径, 用于将草图的 id 对应到 Rico 图片 id
        '''
        super(SynZ, self).__init__()
        
        # settings
        self.pretrained_model_name = pretrained_model_name
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        self.aspect_grouping = aspect_grouping
        self.transform = transform
        self.test_mode = test_mode
        
        # input paths
        self.synz_annotations_path = synz_annotations_path  # json file
        self.prediction_path = prediction_path  # pth file
        self.description_path = description_path  # csv file
        self.sketch_root_path = sketch_root_path    # path to dir of sketches
        
        # load SynZ annotations file in json format
        # annotations are used as ground truth
        # categories are also included
        self.synz_annotations, self.categories, self.rico_id_to_sketch_id = self.load_synz_annotations()
        self.category_map = {category['name']: category['id'] for category in self.categories}
        self.id_to_category = {category['id']: category['name'] for category in self.categories}

        # load predictions generated by SynZ detector
        self.predictions = self.load_predictions()
               
        # load description text from Screen2Words, as DataFrame
        self.descriptions = self.load_descriptions()
        self.descriptions_rico_ids = sorted(self.descriptions['screenId'].unique())
        
        self.avail_rico = [x for x in self.descriptions_rico_ids if x in self.rico_id_to_sketch_id.keys()]
        
        self.cache_dir = None
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)        
        
        self.database = self.build_database()
        
    def build_database(self) -> list:
        '''
            读取数据文件, 构建数据集
            
            由于一张草图可能对应多条文本描述, 每条文本描述视为单独的一条数据
        '''
        database = []
        
        for rico_id in self.avail_rico:
            for sketch_id in self.rico_id_to_sketch_id[rico_id]:
                sketch = self.synz_annotations[sketch_id]
                predictions = self.predictions[sketch_id]
                description = self.descriptions.query(f"screenId == {rico_id}")
                for index, row in description.iterrows():
                    idb = {
                        'sketch_id': sketch_id,
                        'rico_id': rico_id,
                        'sketch_file_name': self.synz_annotations[sketch_id]['file_name'],
                        'width': sketch['width'],
                        'height': sketch['height'],
                        'pred_box': predictions,
                        'description': row['summary'],
                        'gt_anno': sketch['annotations']
                    }
                    database.append(idb)
            
        return database
    
    def load_synz_annotations(self):
        '''
            读取 SynZ 的 annotations json 文件
            
            json 文件中使用到的 keys 有: 'categories', 'images', 'annotations'
            
            'categories' 直接读取返回, 记录了全部的元素类别及对应的 id
            
            'images' 包含了全部的草图, 'annotations' 依次加入到所属的 image 中
        '''
        annotations = {}
        categories = []
        rico_id_to_sketch_id = {}
        with open(self.synz_annotations_path) as ann_file:
            ann = json.load(ann_file)
            categories.extend(ann['categories'])
            
            annotations = {image['id']: image for image in ann['images']}
            for sketch_id, sketch in annotations.items():
                rico_id = int(sketch['file_name'][:-4].split('_')[1])
                if rico_id in rico_id_to_sketch_id.keys():
                    rico_id_to_sketch_id[rico_id].append(sketch_id)
                else:
                    rico_id_to_sketch_id[rico_id] = [sketch_id]
            for annotation in ann['annotations']:
                if annotation['image_id'] not in annotations:
                    print('find annotation id {} for image {}, but image {} does not exist.'.format(annotation['id'], annotation['image_id'], annotation['image_id']))
                else:
                    if 'annotations' not in annotations[annotation['image_id']]:
                        annotations[annotation['image_id']]['annotations'] = []
                    annotations[annotation['image_id']]['annotations'].append(annotation)
        
        return annotations, categories, rico_id_to_sketch_id
                
    def load_predictions(self) -> dict:
        '''
            读取 SynZ 的草图识别结果
        '''
        predictions_raw: list = torch.load(self.prediction_path)
        predictions = {pred['image_id']: pred['instances'] for pred in predictions_raw}
        
        return predictions
    
    def load_descriptions(self):
        '''
            读取 Screen2Words 的描述文本
        '''
        descriptions = pd.read_csv(self.description_path)
        
        return descriptions

    @property
    def data_names(self):
        return ['image', 'boxes', 'image_info', 'description', 'labels', 'gt_labels']
    
    def __len__(self) -> int:
        return len(self.database)
    
    def __getitem__(self, index: int):
        idb = self.database[index]
        
        # sketch image related
        sketch_image_path = os.path.join(self.sketch_root_path, idb['sketch_file_name'])
        sketch_image = self._load_image(sketch_image_path)
        sketch_image_info = torch.as_tensor([idb['width'], idb['height'], 1.0, 1.0])
        
        pred_boxes = []
        pred_labels = []
        for box in idb['pred_box']:
            x_, y_, w_, h_ = box['bbox']
            pred_boxes.append([x_, y_, x_ + w_, y_ + h_])
            pred_labels.append(box['category_id'])
        pred_boxes = torch.as_tensor(pred_boxes)
        pred_labels = torch.as_tensor(pred_labels)
        
        gt_boxes = []
        gt_labels = []
        for box in idb['gt_anno']:
            x_, y_, w_, h_ = box['bbox']
            gt_boxes.append([x_, y_, x_ + w_, y_ + h_])
            gt_labels.append(box['category_id'])
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        gt_labels = torch.as_tensor(gt_labels)
        
        # assign label to each box by its IoU with gt_box
        paired_gt_labels = pair_bbox(pred_boxes, gt_boxes, gt_labels)
        assert pred_labels.size() == paired_gt_labels.size(), 'Size of labels should be same!'
        
        if self.add_image_as_a_box:
            w0, h0 = sketch_image_info[0], sketch_image_info[1]
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            pred_boxes = torch.cat((image_box, pred_boxes), dim=0)
        
        flipped = False
        if self.transform is not None:
            sketch_image, pred_boxes, _, sketch_image_info, flipped = self.transform(sketch_image, pred_boxes, None, sketch_image_info, flipped)
        
        # clamp boxes
        w = sketch_image_info[0].item()
        h = sketch_image_info[1].item()
        pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        
        # description
        descrip_tokens = self.tokenizer.tokenize(idb['description'])
        if flipped:
            descrip_tokens = self.flip_tokens(descrip_tokens, verbose=True)
        descrip_id = self.tokenizer.convert_tokens_to_ids(descrip_tokens)
        
        return sketch_image, pred_boxes, sketch_image_info, descrip_id, pred_labels, paired_gt_labels
    
    def _load_image(self, path):
        return Image.open(path).convert('RGB')
    
    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new
 

def pair_bbox(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, gt_labes: torch.Tensor) -> torch.Tensor:
    match_quality_matrix = pairwise_iou(gt_boxes, pred_boxes)
    matched_vals, matches = match_quality_matrix.max(dim=0)
    pair_labes = gt_labes[matches]
    return pair_labes


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = cal_box_area(boxes1)  # [N]
    area2 = cal_box_area(boxes2)  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou   

def pairwise_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection

def cal_box_area(box: torch.Tensor) -> torch.Tensor:
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area