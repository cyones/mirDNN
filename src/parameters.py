import torch as tr
import getopt
import math
import sys

help_message = "model_fit.py -i <input_fastas> -m <model> -b <batch_size> " \
        "[-d device] [-s seq_length] [-M max_epochs] [-e early_stop] [-l logfile>]" \
        "[-f <focal_loss>]\n\n" \
        "mfasta2csv.py -i <input_fasta> -m <model_file>" \
        "-o <out_csv_file> [-d device]" \
        "model_eval.py -i <input_fasta> -m <model_file>" \
        "-n <number_of_sequences> [-d device]"


class ParameterParser():
    def __init__(self, argv):
        self.input_files = []
        self.output_file = []
        self.model_file = 'model.pmt'
        self.logfile = 'model.log'
        self.random_seed = 42
        self.seq_len = 160
        self.width = 64
        self.n_resnets = 3
        self.kernel_size = 3
        self.upsample = False
        self.batch_size = 1024
        self.valid_prop = 0.1
        self.device = tr.device('cpu')
        self.max_nbatch = float('inf')
        self.early_stop = 100
        self.focal_loss = True
        try:
            opts, args = getopt.getopt(argv, "hi:p:o:m:b:s:l:d:M:e:v:r:f:u:",
                    ["input_fasta=", "output_file=",
                     "model=", "batch_size=", "seq_length=", "logfile=",
                     "device=", "max_epochs=","early_stop=", "valid_prop=",
                     "random_seed=", "focal_loss=", "upsample="])
        except getopt.GetoptError:
            print(help_message)
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print(help_message)
                sys.exit()
            elif opt in ("-i", "--input_fasta"):
                self.input_files.append(arg)
            elif opt in ("-o", "--output_file"):
                self.output_file.append(arg)
            elif opt in ("-m", "--model"):
                self.model_file = arg
            elif opt in ("-b", "--batch_size"):
                self.batch_size = int(arg)
            elif opt in ("-s", "--seq_length"):
                self.seq_len = int(arg)
            elif opt in ("-l", "--logfile"):
                self.logfile = arg
            elif opt in ("-d", "--device"):
                self.device = tr.device(arg)
            elif opt in ("-M", "--max_epochs"):
                self.max_epochs = int(arg)
            elif opt in ("-e", "--early_stop"):
                self.early_stop = int(arg)
            elif opt in ("-v", "--valid_prop"):
                self.valid_prop = float(arg)
            elif opt in ("-f", "--focal_loss"):
                self.focal_loss = bool(arg)
            elif opt in ("-u", "--upsample"):
                self.upsample = bool(arg)
            elif opt in ("-r", "--random_seed"):
                try:
                    self.random_seed = int(arg)
                except ValueError:
                    self.random_seed = None

