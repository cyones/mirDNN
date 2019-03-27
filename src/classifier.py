import torch.nn as nn
from src.fully_connected import FullyConnected

class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(Classifier,self).__init__()
        self.in_dim = in_dim
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, n_classes))
        self.last_layer = nn.Sequential(*layers)
        self.out_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.last_layer(x.view(-1, self.in_dim))
        x = self.out_activation(x)
        return x

