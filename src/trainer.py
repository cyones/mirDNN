import random as rn
import torch as tr
from torch.autograd import Variable
from src.multifasta import Multifasta, nVars


class Trainer():
    def __init__(self, pp):
        self.nbatch = 0
        self.seq_len = pp.seq_len
        self.batch_sizes = pp.batch_sizes
        self.early_stop = pp.early_stop
        self.model_file = pp.model_file
        self.train_loss = 100
        self.valid_loss = 100
        self.best_valid_loss = 100
        self.last_improvement = 0
        self.use_valid_part = pp.valid_prop > 0
        self.finish = False
        self.mfasta = []
        for iff in range(len(pp.input_files)):
            self.mfasta.append(Multifasta(pp.input_files[iff], self.seq_len, pp.valid_prop))
        batch_total = sum(self.batch_sizes)
        self.seq = tr.LongTensor(batch_total, self.seq_len)
        self.val = tr.FloatTensor(batch_total, nVars)
        self.lab = tr.LongTensor(batch_total)

    def load_batch(self, train):
        dst0 = 0
        for iff in range(len(self.batch_sizes)):
            nseqs = self.mfasta[iff].get_nseqs(train)
            if self.batch_sizes[iff] < nseqs:
                sel = rn.sample(range(nseqs), self.batch_sizes[iff])
            else:
                sel = range(nseqs)
            dst1 = dst0 + len(sel)
            self.seq[dst0:dst1] = tr.stack([self.mfasta[iff].get_tns(s, train) for s in sel])
            self.val[dst0:dst1] = tr.stack([self.mfasta[iff].get_mfe(s, train) for s in sel])
            self.lab[dst0:dst1] = iff
            dst0 = dst1

    def train(self, model):
        self.nbatch += 1

        self.load_batch(train=True)
        new_loss = model.train_step(self.seq, self.val, self.lab)
        self.train_loss = new_loss * 0.1 + self.train_loss * 0.9
        if self.use_valid_part:
            self.load_batch(train=False)
            new_loss = model.valid_step(self.seq, self.val, self.lab)
            self.valid_loss = new_loss * 0.1 + self.valid_loss * 0.9
        else:
            self.valid_loss = self.train_loss

        self.last_improvement += 1
        if self.valid_loss < self.best_valid_loss:
            self.best_valid_loss = self.valid_loss
            self.last_improvement = 0
            if self.nbatch > 10:
                tr.save(model.state_dict(), self.model_file)
        self.finish = self.last_improvement > self.early_stop
