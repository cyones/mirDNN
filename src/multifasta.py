import torch as tr
import random as rn
import re
from Bio import SeqIO

nVars = 1

class Multifasta():
    def __init__(self, input_file, seq_len, test_proportion=None):
        self.seq_len = seq_len
        self.gen_code()
        self.nme = []
        self.tns = []
        self.mfe = []
        with open(input_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                rec = str(record.seq.lower().transcribe())
                seq = re.search('[acgu]+', rec).group(0)
                stc = re.search('[\(\.\)]+', rec).group(0)
                mfe = float(re.search('\-?\d+\.\d+', rec).group(0))
                ob, oe, db, de = self.get_limits(len(seq), self.seq_len)
                tns = tr.zeros(self.seq_len, dtype=tr.uint8)
                dg = tr.Tensor([mfe / len(seq)])
                tns[db:de] = tr.ByteTensor([self.seq2num[c] for c in seq[ob:oe]]) + \
                             15 * tr.ByteTensor([self.stc2num[c] for c in stc[ob:oe]])
                self.nme.append(record.id)
                self.mfe.append(dg)
                self.tns.append(tns)
        self.nseq = len(self.mfe)
        if not test_proportion is None:
            self.test_idx = rn.sample(range(self.nseq), int(test_proportion * self.nseq))
            self.train_idx = list(set(list(range(self.nseq))) - set(self.test_idx))

    def get_tns(self, i, train):
        if train:
            return self.tns[self.train_idx[i]]
        else:
            return self.tns[self.test_idx[i]]

    def get_mfe(self, i, train):
        if train:
            return self.mfe[self.train_idx[i]]
        else:
            return self.mfe[self.test_idx[i]]

    def get_nseqs(self, train):
        if train:
            return len(self.train_idx)
        else:
            return len(self.test_idx)

    def get_limits(self, olen, dlen):
            ob = max(int((olen - dlen)/2), 0)
            oe = min(ob + dlen, olen)
            db = max(int((dlen - olen)/2), 0)
            de = min(db + olen, dlen)
            return ob, oe, db, de

    def gen_code(self):
        self.seq2num = {'n': 0, 'a': 1, 'c': 2, 'g': 3, 'u': 4,
                        'w': 5, 's': 6, 'k': 7, 'm': 8, 'y': 9,
                        'r': 10, 'b': 11, 'd': 12, 'h': 13, 'v': 14}
        self.stc2num = {'(': 2, '.': 1, ')': 2}
