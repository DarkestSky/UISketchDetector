from collections import namedtuple
import torch
from common.trainer import to_cuda
from torch.nn import functional as F


@torch.no_grad()
def do_validation(net, val_loader, label_index_in_batch, epoch_num):
    net.eval()
    # metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        logits, boxes = net(*datas)
        
        probs = F.softmax(logits, dim=-1)
        _, y = torch.topk(probs, k=1, dim=-1)
        layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
        recon_layouts = [self.train_dataset.render(layout) for layout in layouts]
        for i, layout in enumerate(layouts):
            layout = self.train_dataset.render(layout)
            layout.save(os.path.join(self.config.samples_dir, f'recon_{epoch:02d}_{i:02d}.png'))
        
        # outputs.update({'label': label})
        # metrics.update(outputs)

