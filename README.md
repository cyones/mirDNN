# MirDNN

MirDNN is a novel deep learning method specifically designed for pre-miRNA prediction in genome-wide data. The model is a convolutional deep residual neural network that can automatically learn suitable features from the raw data, without manual feature engineering.
This model is capable of that can successfully learn the intrinsic structural characteristics of precursors of miRNAs, as well as their context in a sequence. The proposal has been tested with several genomes of animals and plants and compared with state-of-the-art algorithms.

MirDNN was described in detail in the work "High precision in microRNA prediction: a novel genome-wide approach based on convolutional deep residual networks" (under review in a refereed journal).

> Contact: [Cristian Yones](mailto:cyones@sinc.unl.edu.ar),
> [sinc(i)](http://fich.unl.edu.ar/sinc/)

## Web server

MirDNN can be used without need of an installation from [this](http://fich.unl.edu.ar/sinc/web-demo/) web server. This server provides two pre-trained models (animals and plants) and can process both individual sequences or fasta files. When making predictions on individual sequences, the server generates also a nucleotide importance graph. Due to computational limitations, the size of the fasta files that can be uploaded is limited.

## Package installation

The latest version of the package can be downloaded from the GitHub [repository](https://github.com/cyones/mirDNN). The exact version used in the paper is allocated in [SourceForge](https://sourceforge.net/projects/sourcesinc/files/mirdnn).

To download from GitHub:

```bash
git clone --recurse-submodules https://github.com/cyones/mirDNN.git
```
After downloading the package (from GitHub or SourceForge), install the dependencies:

```bash
cd mirDNN
pip install -r requeriments.txt
```

That would install all the needed packages to run mirDNN, but in order to train models or make predictions the secondary structure of the sequences has to be infered. For this task, the ViennaRNA software should be use. To install this software in you OS, see [this](https://www.tbi.univie.ac.at/RNA/) page.

## Usage

To make predictions or training new models, the first step is to predict the secondary structure of the sequences to proccess. This should be done with the RNAfold software. For example, given a fasta file named sequences.fa, run:

```bash
RNAfold --noPS --infile=sequences.fa --outfile=sequences.fold
```

### Inference

Now that we have the *.fold* file, to make predictions with the [provided pre-trained model](./models) for animals, simply run:

```bash
mirdnn_eval -i sequences.fold \
            -o predictions.csv \ # output file
            -m models/animal.pmt \ # pre-trained model provided
            -s 160 \ # sequence max lenght (should be 160 for animal and 320 for plants)
            -d "cpu" # device to use, could be "cpu", "cuda:0", "cuda:1", etc.
```

To calculate nucleotide importance values the command is similar:

```bash
mirdnn_explain -i sequences.fold \
               -o importance.csv \
               -m models/animal.pmt \
               -s 160 \
               -d "cpu"
```

### Training new models

To train new models, two *.fold* files would be needed, one with negative examples (non pre-miRNA sequences) and other with positive examples (well-known pre-miRNAs). For some ideas about how to construct this datasets, see the paper.

Given these datasets, the training can be done with

```bash
mirdnn_fit.py -i negative_sequences.fold \
              -i positive_sequences.fold \
              -m out_model.pmt \
              -l train.log \ # log file with training progress
              -d "cuda:0" \
              -s 160
```

> NOTE: training a model is a very computing intensive task, therefore, it is recommended to use a GPU.

For more details about the training parameters, execute

```bash
mirddn_fit.py -h
```

## Reproduce experiments

All the experiments presented in the paper can be easily reproduced using the Makefile inside the folder [experiments](./experiments). For example, to generate the PRROC curve obtained in *Caenorhabditis elegans*, run:

```bash
cd experiments
make results/PRROC-cel.pdf
```

You would be asked to download the sequences files, and then all the necesary commands to train and test the model would be automatically executed.
