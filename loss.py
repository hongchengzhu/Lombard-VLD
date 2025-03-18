'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))     # cos \theta
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))    # sin \\theta
        phi = cosine * self.cos_m - sine * self.sin_m   # cos(\theta + m) = cos \theta * cos m - sin \theta * sin m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        # convert labels to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
    
    def forward_confusion_matrix(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))     # cos \theta
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))    # sin \\theta
        phi = cosine * self.cos_m - sine * self.sin_m   # cos(\theta + m) = cos \theta * cos m - sin \theta * sin m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        # convert labels to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        tar, far, trr, frr = confusion_matrix(output.detach(), label.detach(), topk=(1,))

        return loss, tar, far, trr, frr