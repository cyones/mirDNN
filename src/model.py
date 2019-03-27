import torch as tr
import torch.nn as nn
import math as mt
import numpy as np
from src.preprocessor import Preprocessor
from src.encoder import Encoder
from src.classifier import Classifier
from src.preprocessor import in_channels
from src.random_shift import RandomShift

class CNN(nn.Module):
    def __init__(self, pp):
        super(CNN, self).__init__()
        self.n_classes = len(pp.input_files)
        self.device = pp.device
        rguess = 1 / self.n_classes
        self.nll_correction = -rguess + mt.log(self.n_classes * mt.exp(rguess))
        self.seq_len = pp.seq_len
        self.preprocessor = Preprocessor(self.device)
        self.encoder = Encoder(in_channels, pp.width, pp.n_resnets, pp.seq_len)
        self.classifier = Classifier(self.encoder.z_dim, self.n_classes)
        self.random_shift = RandomShift(pp.max_shift, pp.device)

        cw = pp.batch_sizes
        if self.n_classes > len(pp.batch_sizes):
            cw = tr.Tensor([1 for i in range(self.n_classes)])
        else:
            cw = tr.Tensor([np.sum(cw) / cw[i] for i in range(len(cw))])
        self.loss_function = nn.NLLLoss(weight=cw)
        self.to(device=self.device)
        self.optimizer = tr.optim.Adadelta(self.parameters())

    def forward(self, x, v):
        x = x.to(device = self.device)
        v = v.to(device = self.device)
        if self.training: x = self.random_shift(x)
        x = self.preprocessor(x)
        z = self.encoder(x, v)
        z = self.classifier(z)
        return z

    def train_step(self, x, v, y):
        self.optimizer.zero_grad()
        z = self(x, v)
        loss = self.loss_function(z, y.to(device = self.device))
        loss.backward()
        self.optimizer.step()
        loss_val = 100 * loss.data.item() / self.nll_correction
        return loss_val

    def valid_step(self, x, v, y):
        z = self(x, v)
        loss = self.loss_function(z, y.to(device = self.device))
        loss_val = 100 * loss.data.item() / self.nll_correction
        return loss_val
