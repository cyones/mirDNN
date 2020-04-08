import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_dim, nfilters, ksizes):
        super(ResNet, self).__init__()
        layers=[]
        for i in range(len(nfilters)):
            npad = int(ksizes[i]/2)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Conv1d(in_dim, nfilters[i],
                kernel_size=ksizes[i], padding=npad))
            in_dim = nfilters[i]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x

