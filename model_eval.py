import sys
import os.path
import torch as tr
import csv

from src.multifasta import Multifasta
from src.model import CNN
from src.parameters import ParameterParser

def main(argv):
    pp = ParameterParser(argv)
    cae = CNN(pp)
    cae.load_state_dict(tr.load(pp.model_file, map_location=lambda storage, loc: storage))
    cae.eval()
    bsize = pp.batch_sizes[0]
    if pp.device.type == 'cuda':
        tr.backends.cudnn.benchmark = True

    for i in range(len(pp.input_files)):
        mf = Multifasta(pp.input_files[i], pp.seq_len)
        N = len(mf.tns)
        ib = 0
        ie = 0
        with open(pp.output_file[i], 'w') as csvfile:
            of = csv.writer(csvfile, delimiter=',', )
            while ie < N:
                ie = min(ib+bsize, N)
                seq = tr.stack([mf.tns[s] for s in range(ib,ie)])
                val = tr.stack([mf.mfe[s] for s in range(ib,ie)])
                seq = seq.to(dtype=tr.int64)
                res = cae(seq, val).data.tolist()
                for k in range(len(res)):
                    line = res[k]
                    line.insert(0, mf.nme[ib + k])
                    of.writerow(line)
                ib = ie

if __name__ == "__main__":
    main(sys.argv[1:])
