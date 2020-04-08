import torch as tr
import random as rn
import re
import gc
from Bio import SeqIO
from torch.utils.data import Dataset

class FoldDataset(Dataset):
    def __init__(self, input_files, seq_len):
        self.seq_len = seq_len
        self.nsamples_class = []
        self.input_files = input_files

        total_seqs = 0
        for label, filename in enumerate(input_files):
            nsamples = 0
            with open(filename, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    nsamples += 1
            self.nsamples_class.append(nsamples)
            total_seqs += nsamples

        self.name = list(range(total_seqs))
        self.sequence = tr.ByteTensor(total_seqs, seq_len)
        self.mfe = tr.Tensor(total_seqs, 1)
        self.label = tr.Tensor(total_seqs, 1)

        for label, filename in enumerate(input_files):
            with open(filename, "r") as handle:
                for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                    self.name[i] = record.id

                    tns, mfe = self.record2tensor(record)
                    self.sequence[i] = tns
                    self.mfe[i] = mfe
                    self.label[i] = label


    def record2tensor(self, record):
        rec = str(record.seq.lower().transcribe())
        seq = re.search('[acgu]+', rec).group(0)
        stc = re.search('[\(\.\)]+', rec).group(0)
        mfe = float(re.search('\-?\d+\.\d+', rec).group(0)) / len(seq)

        tns = tr.ByteTensor([self.seq2num[n] + 15 * self.stc2num[s] + 1
            for n, s in zip(seq, stc)])
        if tns.shape[0] < self.seq_len:
            pad = self.seq_len - tns.shape[0]
            lpad = int(pad / 2)
            rpad = pad - lpad
            tns = tr.nn.functional.pad(tns, (lpad, rpad))
        if tns.shape[0] > self.seq_len:
            pad = tns.shape[0] - self.seq_len
            llim = int(pad / 2)
            rlim = self.seq_len + llim
            tns = tns[llim:rlim]

        return tns, mfe


    def __getitem__(self, index):
        seq = self.sequence[index].long()
        mfe = self.mfe[index]
        label = self.label[index]
        return seq, mfe, label

    def __len__(self):
        return len(self.sequence)

    seq2num = {'a':  0, 'c':  1, 'g':  2, 'u':  3, 'w':  4,
               's':  5, 'k':  6, 'm':  7, 'y':  8, 'r':  9,
               'b': 10, 'd': 11, 'h': 12, 'v': 13, 'n': 14}
    stc2num = {'.': 0, '(': 1, ')': 2}
