import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, in_dim, nunits, dout):
        super(FullyConnected, self).__init__()
        self.in_dim = in_dim
        layers = []
        for i in range(len(nunits)):
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Linear(in_dim, nunits[i]))
            if dout[i]: layers.append(nn.Dropout(0.2))
            in_dim = nunits[i]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x.view(-1, self.in_dim))

