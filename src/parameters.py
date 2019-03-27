import torch as tr
import getopt
import math
import sys

help_message = "model_fit.py -i <input_fastas> -b <batch_sizes> -m <model>" \
        "[-d device] [-s seq_length] [-M max_nbatch] [-e early_stop] [-L logfile>]\n\n" \
        "mfasta2csv.py -i <input_fasta> -m <model_file>" \
        "-o <out_csv_file> [-g gpu_device_number]" \
        "model_eval.py -i <input_fasta> -m <model_file>" \
        "-n <number_of_sequences> [-g gpu_device_number]"


class ParameterParser():
    def __init__(self, argv):
        self.input_files = []
        self.batch_sizes = []
        self.output_file = []
        self.model_file = 'model.pmt'
        self.logfile = 'model.log'
        self.number_seq = 3
        self.random_seed = 42
        self.seq_len = 160
        self.width = 32
        self.n_resnets = 3
        self.max_shift = 10
        self.valid_prop = 0.1
        self.device = tr.device('cpu')
        self.max_nbatch = math.inf
        self.early_stop = 100
        try:
            opts, args = getopt.getopt(argv, "hi:b:o:m:s:f:d:M:n:e:l:L:v:r:",
                    ["input_fasta=", "batch_size=", "output_file=", "model=", "seq_length=",
                        "logfile=", "device=", "max_nbatch=", "number_seq=", "early_stop=",
                        "valid_prop=", "random_seed="])
        except getopt.GetoptError:
            print(help_message)
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print(help_message)
                sys.exit()
            elif opt in ("-i", "--input_fasta"):
                self.input_files.append(arg)
            elif opt in ("-b", "--batch_size"):
                self.batch_sizes.append(int(arg))
            elif opt in ("-o", "--output_file"):
                self.output_file.append(arg)
            elif opt in ("-m", "--model"):
                self.model_file = arg
            elif opt in ("-s", "--seq_length"):
                self.seq_len = int(arg)
                self.max_shift = int(self.seq_len / 16)
            elif opt in ("-r", "--random_seed"):
                try:
                    self.random_seed = int(arg)
                except ValueError:
                    self.random_seed = None
            elif opt in ("-d", "--device"):
                self.device = tr.device(arg)
            elif opt in ("-t", "--valid_prop"):
                self.valid_prop = float(arg)
            elif opt in ("-n", "--number_seq"):
                self.number_seq = int(arg)
            elif opt in ("-M", "--max_nbatch"):
                self.max_nbatch = int(arg)
            elif opt in ("-e", "--early_stop"):
                self.early_stop = int(arg)
            elif opt in ("-L", "--logfile"):
                self.logfile = arg

