'''
    Do validation, and remember the best epoch
'''

import logging
import shutil

from numpy import Inf

class ValidationMonitor(object):
    def __init__(self, val_func, val_loader, metrics, host_metric_name='Acc', label_index_in_batch=-1):
        super(ValidationMonitor, self).__init__()
        self.val_func = val_func
        self.val_loader = val_loader
        self.metrics = metrics
        self.host_metric_name = host_metric_name
        self.loss_mean = Inf
        self.best_epoch = -1
        self.best_val = -1.0
        self.label_index_in_batch = label_index_in_batch

    def state_dict(self):
        return {'best_epoch': self.best_epoch,
                'best_val': self.best_val}

    def load_state_dict(self, state_dict):
        assert 'best_epoch' in state_dict, 'miss key \'best_epoch\''
        assert 'best_val' in state_dict, 'miss key \'best_val\''
        self.best_epoch = state_dict['best_epoch']
        self.best_val = state_dict['best_val']

    def __call__(self, epoch_num, net, optimizer, writer):
        loss_mean = self.val_func(net, self.val_loader, self.label_index_in_batch, epoch_num)

        if loss_mean < self.loss_mean:
            self.best_epoch = epoch_num
            self.loss_mean = loss_mean
            logging.info('New Best Val loss: {}, Epoch: {}'.format(self.loss_mean, self.best_epoch))
            print('New Best Val loss: {}, Epoch: {}'.format(self.loss_mean, self.best_epoch))
        if writer is not None:
            writer.add_scalar(tag='Val',
                              scalar_value=loss_mean,
                              global_step=epoch_num + 1)
 
