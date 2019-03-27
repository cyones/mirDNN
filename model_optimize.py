import os.path
import sys
import time
import torch as tr
import numpy as np
import random as rn

from src.trainer import Trainer
from src.model import CNN
from src.parameters import ParameterParser
from src.logger import Logger

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
    for nr in [1, 2, 3]:
        for wd in [16, 32, 64]:
            pp.n_resnets = nr
            pp.width = wd
            trainer = Trainer(pp)
            cae = CNN(pp)
            cae.train()
            log = Logger(pp.logfile)

            log.write('nResnets\twidth\tnbatch\ttrainL\tvalidL\telapsed\tthroughput\n')
            tstart = time.time()
            while trainer.nbatch < pp.max_nbatch:
                trainer.train(cae)
                elapsed = time.time()-tstart
                throughput = sum(pp.batch_sizes) / elapsed / 1000
                log.write('%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                      (pp.n_resnets, pp.width, trainer.nbatch,
                          trainer.train_loss, trainer.valid_loss, elapsed, throughput))
                tstart = time.time()
                if trainer.finish: break
            del trainer, cae
    log.close()

if __name__ == "__main__":
    main(sys.argv[1:])
