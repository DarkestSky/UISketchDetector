from collections import namedtuple
import numpy as np
import torch
from common.trainer import to_cuda
from torch.nn import functional as F


@torch.no_grad()
def do_validation(net, val_loader, label_index_in_batch, epoch_num):
    '''
        在验证集上检测模型效果
        使用 loss 的均值作为标准
    '''
    net.eval()
    losses = []
    for nbatch, batch in enumerate(val_loader):            
        batch = to_cuda(batch)
        
        logits, boxes, loss = net(*batch)
        loss = loss.mean().cpu().detach().numpy()
        losses.append(loss)
    
    return float(np.mean(losses))
