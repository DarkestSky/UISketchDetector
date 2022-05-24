import os
import pprint
import shutil
import random

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageOps
import seaborn as sns

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from refcoco.data.build import make_dataloader
from refcoco.modules import *

POSITIVE_THRESHOLD = 0.5


def cacluate_iou(pred_boxes, gt_boxes):
    x11, y11, x12, y12 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x21, y21, x22, y22 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = (xB - xA + 1).clip(0) * (yB - yA + 1).clip(0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


@torch.no_grad()
def test_net(args, config):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    #os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
    config.DATASET.TEST_IMAGE_SET = args.split
    ckpt_path = args.ckpt
    save_path = args.result_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    colors = gen_colors(21)

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='test', distributed=False)
    sketch_id_list = test_loader.dataset.sketch_id_list

    # test
    pred_boxes = []
    model.eval()
    # cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
    # for nbatch, batch in tqdm(enumerate(test_loader)):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        # ref_ids.extend([test_database[id]['ref_id'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        batch = to_cuda(batch)
        logits, boxes, _ = model(*batch)
                
        probs = F.softmax(logits, dim=-1)
        value, index = torch.topk(probs, k=1, dim=-1)
        
        for score_list, id_list, box_list in zip(value, index, boxes):
            idx = {
                'pred_boxes': box_list,
                'scores': score_list,
                'pred_classes': id_list
            }
            pred_boxes.append(idx)
        
        # boxes_list = boxes.numpy().tolist()
        
        if (random.randint(0, 9) < 2):
            sample_path = os.path.join(save_path, 'sample/')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            for i, (cls, box) in enumerate(zip(index, boxes)):
                img = render(cls, box, colors)
                sketch_id = sketch_id_list[i+bs*nbatch]
                img.save(os.path.join(save_path, 'sample/', f'{sketch_id}.png'))

    result = {
        'image_id': sketch_id_list,
        'instance': pred_boxes
    }
    
    pth_path = os.path.join(save_path, 'pth/')
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    result_path = os.path.join(pth_path, 'result.pth')
    torch.save(result, result_path)

    return result_path


def render(cls, box, colors):
    img = Image.new('RGB', (576, 1024), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    draw_list = seletc_box(box)
    box = box.cpu().detach().numpy().tolist()
    
    for i in draw_list:
        x1, y1, x2, y2 = box[i]
        cat = cls[i]

        col = colors[cat] 
        draw.rectangle([x1, y1, x2, y2],
                    outline=tuple(col) + (200,),
                    fill=tuple(col) + (64,),
                    width=2)

            # font_path = 'font/Lantinghei.ttc'
            # font = ImageFont.truetype(font_path, 40)
            # draw.text((x1+5,y1+5),self.contiguous_category_id_to_csv[cat], tuple(col),font)
    # Add border around image
    img = ImageOps.expand(img, border=2)
    return img


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    [num_colors, 3]
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


def seletc_box(boxes):
    iou = pairwise_iou(boxes, boxes)
    area = cal_box_area(boxes)
    res_list = []
    for i in range(boxes.shape[0]):
        if area[i] > 500 and iou[i][res_list].sum() < 0.2:
            res_list.append(i)
    return res_list


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