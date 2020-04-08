import torch as tr
from torch import nn

class FocalLoss(tr.nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCEloss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = tr.exp(-BCEloss)
        Floss = (1-pt)**self.gamma * BCEloss
        return tr.mean(Floss)
