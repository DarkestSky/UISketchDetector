'''
    基于 VL-BERT 提供的在 RefCOCO+ 数据集上运行的模型修改
    模型运行在新的数据集上
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform
from .gpt import GPTConfig, GPT

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)
        self.add_image_as_a_box = config.DATASET.ADD_IMAGE_AS_A_BOX     # 图片整体也作为一个 box 传入

        # 使用 Faster R-CNN 提取各个区域的特征
        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        #####  NOT USED #####
        
        self.gpt_bos_token = config.NETWORK.GPT_VOCAB_SIZE - 3
        self.gpt_eos_token = config.NETWORK.GPT_VOCAB_SIZE - 2
        self.gpt_pad_token = config.NETWORK.GPT_VOCAB_SIZE - 1
        self.gpt_max_length = config.NETWORK.GPT_BLOCK_SIZE
        
        ######################
        
        # GPT 接收的序列长度为 50, 不使用 bos eos 及 pad 字符
        gpt_config = GPTConfig(config.NETWORK.GPT_VOCAB_SIZE, config.NETWORK.GPT_BLOCK_SIZE, 
                               n_layer = config.NETWORK.GPT_N_LAYER, n_head = config.NETWORK.GPT_N_HEAD, n_embd = config.NETWORK.GPT_N_EMBD)
        self.gpt = GPT(gpt_config)
        
        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        # GPT 中的参数在 GPT 的 init 方法中进行初始化

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      pred_labels,
                      gt_labels,
                      orig_boxes,
                      ):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        pred_labels = pred_labels[:, :max_len]
        gt_labels = gt_labels[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, _ = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=True)

        ###########################################
                
        # generator
        logits, loss = self.gpt(pred_labels, hidden_states_regions, gt_labels, pad_token = self.gpt_pad_token)

        return logits, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression,
                          pred_labels,
                          gt_labels,
                          orig_boxes,
                          ):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, _ = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=True)

        ###########################################
        
        # generator
        logits, loss = self.gpt(pred_labels, hidden_states_regions, gt_labels, pad_token = self.gpt_pad_token)
        
        return logits, orig_boxes, loss

