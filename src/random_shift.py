import torch as tr
import random as rn

class RandomShift(tr.nn.Module):
    def __init__(self, max_shift, device):
        super(RandomShift, self).__init__()
        self.max_shift = max_shift
        self.device = device

    def forward(self, x):
        shift = rn.randint(-self.max_shift, self.max_shift)
        if shift > 0:
            pad = tr.zeros(x.size(0),  shift, device = self.device, dtype=tr.long)
            x = tr.cat([x[:,shift:], pad], dim=1)
        elif shift < 0:
            pad = tr.zeros(x.size(0), -shift, device = self.device, dtype=tr.long)
            x = tr.cat([pad, x[:,:shift]], dim=1)
        return x


