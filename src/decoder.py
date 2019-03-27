import torch.nn as nn
import math as mt
from src.resnet import ResNet
from src.fully_connected import FullyConnected

class Decoder(nn.Module):
    def __init__(self, in_deep, seq_len):
        super(Decoder, self).__init__()
        nn_factor = 4/3
        self.conv_depth = 3
        self.seq_len = seq_len

        n_fc_layers = int(mt.log(self.conv_depth * self.seq_len / in_deep) / mt.log(nn_factor))
        nu = [int(in_deep*nn_factor**i) for i in range(n_fc_layers)]
        nu.append(self.conv_depth * self.seq_len)
        do = [True for i in range(len(nu)-1)]
        do.append(False)
        self.fcon_layers = FullyConnected(in_deep, nu, do)

        layers = []
        layers.append(ResNet(self.seq_len, nfilters=[self.conv_depth*2, self.conv_depth], ksizes=[5, 5]))
        layers.append(ResNet(self.seq_len, nfilters=[self.conv_depth*2, self.conv_depth], ksizes=[5, 5]))
        layers.append(ResNet(self.seq_len, nfilters=[self.conv_depth*2, self.conv_depth], ksizes=[5, 5]))
        layers.append(nn.Conv1d(self.conv_depth, 5, kernel_size=5, padding=2))
        layers.append(ResNet(self.seq_len, nfilters=[10, 5], ksizes=[5, 5]))
        layers.append(ResNet(self.seq_len, nfilters=[10, 5], ksizes=[5, 5]))
        layers.append(ResNet(self.seq_len, nfilters=[10, 5], ksizes=[5, 5]))
        layers.append(nn.Sigmoid())
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fcon_layers(x)
        x = self.conv_layers(x)
        return x

