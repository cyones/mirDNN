import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, seq_len, nfilters, ksizes):
        super(ResNet, self).__init__()
        self.seq_len = seq_len
        self.in_dim = nfilters[len(nfilters)-1]
        nfilters.insert(0, self.in_dim)
        layers=[]
        for i in range(len(nfilters)-1):
            npad = int(ksizes[i]/2)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(nfilters[i]))
            layers.append(nn.Conv1d(nfilters[i], nfilters[i+1], kernel_size=ksizes[i], padding=npad))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(-1, self.in_dim, self.seq_len)
        return self.layers(x) + x

