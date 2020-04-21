import torch as tr
import torch.nn as nn
import math as mt
import numpy as np
from src.embedding import NucleotideEmbedding, in_channels
from src.focal_loss import FocalLoss
from src.resnet import ResNet
from src.RAdam.radam import RAdam

class mirDNN(nn.Module):
    def __init__(self, pp):
        super(mirDNN, self).__init__()
        self.device = pp.device
        self.nll_correction = -0.5 + mt.log(2 * mt.exp(0.5))

        self.embedding = NucleotideEmbedding()
        layers = []
        layers.append(nn.Conv1d(in_channels,
                                pp.width,
                                kernel_size=pp.kernel_size,
                                padding=int(pp.kernel_size/2)))
        seq_len = pp.seq_len
        while seq_len > 10:
            for i in range(pp.n_resnets):
                layers.append(ResNet(pp.width,
                                     nfilters=[pp.width, pp.width],
                                     ksizes=[pp.kernel_size, pp.kernel_size]))
            layers.append(nn.MaxPool1d(2))
            seq_len = int(seq_len / 2)
        layers.append(nn.ELU())
        layers.append(nn.BatchNorm1d(pp.width))
        self.conv_layers = nn.Sequential(*layers)
        self.conv_out_dim = pp.width * seq_len

        self.ivar_layers = nn.BatchNorm1d(1)

        in_dim = self.conv_out_dim + 1
        layers = []
        layers.append(nn.Linear(in_dim, 32))
        layers.append(nn.ELU())
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.Linear(32, 1))
        layers.append(nn.Sigmoid())
        self.fcon_layers = nn.Sequential(*layers)

        if pp.focal_loss:
            self.loss_function = FocalLoss()
        else:
            self.loss_function = nn.BCELoss()
        self.to(device = self.device)
        self.optimizer = RAdam(self.parameters(), lr=5e-3, weight_decay=1e-5)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                              patience=100, min_lr=1e-6, eps=1e-9)

    def forward(self, x, v):
        x = x.to(device = self.device)
        v = v.to(device = self.device)

        v = self.ivar_layers(v)
        x = self.embedding(x)
        x = self.conv_layers(x)
        x = x.view(-1, self.conv_out_dim)
        x = tr.cat([x, v], dim=1)
        x = self.fcon_layers(x)
        return x

    def train_step(self, x, v, y):
        self.optimizer.zero_grad()
        z = self(x, v)
        loss = self.loss_function(z, y.to(device = self.device))
        loss.backward()
        self.optimizer.step()
        loss_val = 100 * loss.data.item() / self.nll_correction
        return loss_val

    def load(self, model_file):
        self.load_state_dict(tr.load(model_file, map_location=lambda storage, loc: storage))

    def save(self, model_file):
        tr.save(self.state_dict(), model_file)
