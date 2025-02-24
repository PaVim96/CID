from __future__ import print_function

import torch
import numpy as np

def normalize( x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized



def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), returnCorrect = False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        lstCorrect = []
        for k in topk:
            try:
                if returnCorrect:
                    lstCorrect.append(correct)
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            except RuntimeError:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        if returnCorrect:
            return res, lstCorrect
        else: 
            return res
    



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def cluster(f_s, f_t, label, num_classes):
    
    length= f_s.shape[0]
    
    list_s = []
    list_t = []
    
    for i in range(num_classes):
        list_s.append([])
        list_t.append([])
        
    for i in range(length):
        list_s[label[i]].append(f_s[i])
        list_t[label[i]].append(f_t[i])
        
        
    return list_s, list_t





if __name__ == '__main__':

    pass
