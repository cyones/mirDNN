import torch as tr
import torch.nn as nn
import math as mt
from src.resnet import ResNet
from src.fully_connected import FullyConnected
from src.multifasta import nVars

class Encoder(nn.Module):
    def __init__(self, in_deep, width, n_resnets, seq_len):
        super(Encoder, self).__init__()
        layers = []
        self.z_dim = 10
        layers.append(nn.Conv1d(in_deep, width, kernel_size=5, padding=2))
        while seq_len>10:
            for i in range(n_resnets):
                layers.append(ResNet(seq_len, nfilters=[width, width], ksizes=[3, 3]))
            layers.append(nn.MaxPool1d(2))
            seq_len = int(seq_len / 2)
        self.conv_layers = nn.Sequential(*layers)
        self.conv_out_dim = width * seq_len

        self.ivar_layers = nn.Sequential(nn.BatchNorm1d(nVars), nn.Linear(nVars, nVars))
        self.fcon_layers = FullyConnected(self.conv_out_dim + nVars, [20, self.z_dim], [True, True])

    def forward(self, x, v):
        x = self.conv_layers(x)
        x = x.view(-1, self.conv_out_dim)
        v = self.ivar_layers(v)
        x = tr.cat([x, v], dim=1)
        x = self.fcon_layers(x)
        return x

