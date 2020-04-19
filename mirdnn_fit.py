#!/usr/bin/python3
import os.path
import sys
import time
import torch as tr
import numpy as np
import random as rn
import torch.utils.data as dt
from sklearn.metrics import precision_recall_curve, auc
from src.fold_dataset import FoldDataset
from src.model import mirDNN
from src.parameters import ParameterParser
from src.sampler import ImbalancedDatasetSampler
from src.logger import Logger

tr.multiprocessing.set_sharing_strategy('file_system')

def main(argv):
    pp = ParameterParser(argv)
    if not pp.random_seed is None:
        rn.seed(pp.random_seed)
        np.random.seed(pp.random_seed)
        tr.manual_seed(pp.random_seed)
    if pp.device.type == 'cuda':
        if not pp.random_seed is None:
            tr.backends.cudnn.deterministic = True
            tr.backends.cudnn.benchmark = False
        else:
            tr.backends.cudnn.deterministic = False
            tr.backends.cudnn.benchmark = True

    dataset = FoldDataset(pp.input_files, pp.seq_len)
    valid_size = int(pp.valid_prop * len(dataset))
    train, valid = dt.random_split(dataset, (len(dataset)-valid_size, valid_size))

    train_loader = None
    if pp.upsample:
        train_sampler = ImbalancedDatasetSampler(train,
                                                 max_imbalance = 1.0,
                                                 num_samples = 8 * pp.batch_size)

        train_loader = dt.DataLoader(train,
                                     batch_size=pp.batch_size,
                                     shuffle=True,
                                     sampler=train_sampler,
                                     pin_memory=True,
                                     num_workers = 2)
    else:
        train_loader = dt.DataLoader(train,
                                     batch_size=pp.batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers = 2)

    valid_loader = dt.DataLoader(valid,
                                 batch_size=pp.batch_size,
                                 pin_memory=True,
                                 num_workers=2)

    model = mirDNN(pp)
    model.train()
    log = Logger(pp.logfile)

    if not pp.model_file is None and os.path.isfile(pp.model_file):
        model.load(pp.model_file)

    log.write('epoch\ttrainLoss\tvalidAUC\tlast_imp\n')
    epoch = 0
    train_loss = 100
    valid_auc = 0
    best_valid_auc = 0
    last_improvement = 0
    while last_improvement < pp.early_stop:
        nbatch = 0
        for x, v, y in train_loader:
            new_loss = model.train_step(x, v, y)
            train_loss = 0.99 * train_loss + 0.01 * new_loss
            nbatch += 1
            if nbatch >= 1000: continue

        preds, labels = tr.Tensor([]), tr.Tensor([])
        for x, v, y in valid_loader:
            z = model(x, v).cpu().detach()
            preds = tr.cat([preds, z.squeeze()])
            labels = tr.cat([labels, y.squeeze()])
        pr, rc, _ = precision_recall_curve(labels, preds)
        valid_auc = 10 * auc(rc, pr) + 0.9 * valid_auc

        model.lr_scheduler.step(valid_auc)
        last_improvement += 1
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            last_improvement = 0
            model.save(pp.model_file)

        log.write('%d\t%.4f\t%.4f\t%d\n' %
                (epoch, train_loss, valid_auc, last_improvement))
        epoch += 1
    log.close()

if __name__ == "__main__":
    main(sys.argv[1:])
