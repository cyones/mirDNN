import sys
import os.path
import torch as tr
from tqdm import tqdm
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
        ind = tr.LongTensor(range(pp.seq_len))
        with open(pp.output_file[i], 'w') as csvfile:
            of = csv.writer(csvfile, delimiter=',', )

            line = ["sequence_name"] + [",score"] + \
                   [",N{0}".format(i) for i in range(pp.seq_len)]
            of.writerow(line)

            for i, data in enumerate(tqdm(dataset)):
                x, v, _ = data

                mean = model(x.unsqueeze(0), v.unsqueeze(0)).cpu().detach().item()

                x = x.repeat(pp.seq_len, 1)
                x[ind,ind] = 0
                v = v.repeat(pp.seq_len, 1)

                z = model(x, v).cpu().detach().squeeze()
                z = mean - z

                line = [dataset.name[i]] + [mean] + z.tolist()
                of.writerow(line)

if __name__ == "__main__":
    main(sys.argv[1:])
