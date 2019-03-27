import torch as tr
import torch.nn as nn
import math as mt

in_channels = 5

class Preprocessor(nn.Module):
    def __init__(self, device):
        super(Preprocessor, self).__init__()
        p = 1 / 2
        n = -p
        self.map = tr.Tensor([[ 0, 0, 0, 0,  0],
                             [  p, n, n, n,  0],
                             [  n, p, n, n,  0],
                             [  n, n, p, n,  0],
                             [  n, n, n, p,  0],
                             [  p, n, n, p,  0],
                             [  n, p, p, n,  0],
                             [  n, n, p, p,  0],
                             [  p, p, n, n,  0],
                             [  n, p, n, p,  0],
                             [  p, n, p, n,  0],
                             [  n, p, p, p,  0],
                             [  p, n, p, p,  0],
                             [  p, p, n, p,  0],
                             [  p, p, p, n,  0],
                             [  0, 0, 0, 0,  1],
                             [  p, n, n, n,  1],
                             [  n, p, n, n,  1],
                             [  n, n, p, n,  1],
                             [  n, n, n, p,  1],
                             [  p, n, n, p,  1],
                             [  n, p, p, n,  1],
                             [  n, n, p, p,  1],
                             [  p, p, n, n,  1],
                             [  n, p, n, p,  1],
                             [  p, n, p, n,  1],
                             [  n, p, p, p,  1],
                             [  p, n, p, p,  1],
                             [  p, p, n, p,  1],
                             [  p, p, p, n,  1],
                             [  0, 0, 0, 0, -1],
                             [  p, n, n, n, -1],
                             [  n, p, n, n, -1],
                             [  n, n, p, n, -1],
                             [  n, n, n, p, -1],
                             [  p, n, n, p, -1],
                             [  n, p, p, n, -1],
                             [  n, n, p, p, -1],
                             [  p, p, n, n, -1],
                             [  n, p, n, p, -1],
                             [  p, n, p, n, -1],
                             [  n, p, p, p, -1],
                             [  p, n, p, p, -1],
                             [  p, p, n, p, -1],
                             [  p, p, p, n, -1]]).to(device)

    def forward(self, x):
        y = self.map[x, :].contiguous()
        y.transpose_(1,2)
        return y

