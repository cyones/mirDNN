#!/usr/bin/python3
import sys
import os.path
import torch as tr
import csv

from torch.utils.data import DataLoader
from src.model import mirDNN
from src.parameters import ParameterParser
from src.fold_dataset import FoldDataset

def main(argv):
    pp = ParameterParser(argv)

    model = mirDNN(pp)
    model.load(pp.model_file)
    model.eval()

    for i, ifile in enumerate(pp.input_files):
        dataset = FoldDataset([ifile], pp.seq_len)
        loader = DataLoader(dataset, batch_size=pp.batch_size, pin_memory=True)
        with open(pp.output_file[i], 'w') as csvfile:
            of = csv.writer(csvfile, delimiter=',', )
            for i, sample in enumerate(loader):
                seq, val, _ = sample
                res = model(seq, val).data.tolist()
                for k, pred in enumerate(res):
                    line = [dataset.name[i * pp.batch_size + k], pred[0]]
                    of.writerow(line)

if __name__ == "__main__":
    main(sys.argv[1:])
