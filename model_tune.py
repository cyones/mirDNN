import os.path
import sys
import time
import torch as tr
import numpy as np
import random as rn
import pandas as pd
import torch.utils.data as dt
from sklearn.metrics import precision_recall_curve, auc
from src.fold_dataset import FoldDataset
from src.model import mirDNN
from src.parameters import ParameterParser
from src.sampler import ImbalancedDatasetSampler
from src.logger import Logger

tr.multiprocessing.set_sharing_strategy('file_system')
num_workers = 4

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

    test_size = int(0.1 * len(dataset))
    train, test = dt.random_split(dataset, (len(dataset)-test_size, test_size))

    valid_size = int(pp.valid_prop * len(train))
    train, valid = dt.random_split(train, (len(train)-valid_size, valid_size))


    if os.path.isfile(pp.output_file[0]):
        results = pd.read_csv(pp.output_file[0])
    else:
        results = pd.DataFrame({'kernel_size' : [],
                                'width' : [],
                                'n_resnets' : [],
                                'focal_loss' : [],
                                'max_imbalance' : [],
                                'best_valid_auc' : []})

    log = Logger(pp.logfile)
    log.write('kerner_size\twidth\tn_resnets\t' +\
              'focal_loss\tmax_imbalance\tbest_valid_auc\n')
    for width in [16, 32, 64]:
        for n_resnets in [1, 3, 5]:
            for focal_loss in [True, False]:
                for max_imbalance in [10, 20, 30]:
                    for kernel_size in [3, 5]:
                        if ((results['width']==width) & \
                            (results['n_resnets']==n_resnets) & \
                            (results['focal_loss']==focal_loss)& \
                            (results['max_imbalance']==max_imbalance)&\
                            (results['kernel_size']==kernel_size)).sum() > 2:
                            continue
                        pp.width = width
                        pp.n_resnets = n_resnets
                        pp.focal_loss = focal_loss
                        pp.kernel_size = kernel_size
                        pp.max_imbalance = max_imbalance

                        best_valid_auc = eval_model(train,valid,test,pp)

                        log.write('%d\t%d\t%d\t%d\t%.1f\t%.4f\n' %
                                (kernel_size, width, n_resnets, focal_loss, \
                                 max_imbalance, best_valid_auc))

                        results = results.append({'kernel_size' : kernel_size,
                                                  'width' : width,
                                                  'n_resnets' : n_resnets,
                                                  'focal_loss' : focal_loss,
                                                  'max_imbalance' : max_imbalance,
                                                  'best_valid_auc' : best_valid_auc},
                                                  ignore_index = True)
                        results.to_csv(pp.output_file[0], index=False)

    log.close()


def eval_model(train, valid, test, pp):
    train_sampler = ImbalancedDatasetSampler(train,
                                             max_imbalance = pp.max_imbalance,
                                             num_samples = 8 * pp.batch_size)

    train_loader = dt.DataLoader(train,
                                 batch_size=pp.batch_size,
                                 sampler=train_sampler,
                                 pin_memory=True,
                                 num_workers = num_workers)

    valid_loader = dt.DataLoader(valid,
                                 batch_size=pp.batch_size,
                                 pin_memory=True,
                                 num_workers = num_workers)

    test_loader = dt.DataLoader(test,
                                batch_size=pp.batch_size,
                                pin_memory=True,
                                num_workers = num_workers)
    model = mirDNN(pp)
    model.train()

    nbatch = 0
    train_loss = 100
    valid_auc = 0
    best_valid_auc = 0
    last_improvement = 0
    while last_improvement < pp.early_stop:
        for x, v, y in train_loader:
            new_loss = model.train_step(x, v, y)
            train_loss = 0.9 * train_loss + 0.1 * new_loss

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

        nbatch += 1

    preds, labels = tr.Tensor([]), tr.Tensor([])
    for x, v, y in test_loader:
        z = model(x, v).cpu().detach()
        preds = tr.cat([preds, z.squeeze()])
        labels = tr.cat([labels, y.squeeze()])
    pr, rc, _ = precision_recall_curve(labels, preds)
    test_auc = 10 * auc(rc, pr) + 0.9 * valid_auc
    return test_auc


if __name__ == "__main__":
    main(sys.argv[1:])
