# MirDNN

MirDNN is a novel deep learning method specifically designed for pre-miRNA prediction in genome-wide data. The model is a convolutional deep residual neural network that can automatically learn suitable features from the raw data, without manual feature engineering.
This model is capable of successfully learn the intrinsic structural characteristics of precursors of miRNAs, as well as their context in a sequence. The proposal has been tested with several genomes of animals and plants and compared with state-of-the-art algorithms.

MirDNN is described with detail in "High precision in microRNA prediction: a novel genome-wide approach based on convolutional deep residual networks," by C. Yones, L.A. Bugnon, J. Raad, D.H. Milone and G. Stegmayer (under review in a refereed journal).

> Contact: [Cristian Yones](mailto:cyones@sinc.unl.edu.ar), [sinc(i)](http://sinc.unl.edu.ar)

## Web server

MirDNN can be used directly from [this](http://sinc.unl.edu.ar/sinc/web-demo/mirdnn) web server. This server provides two mirDNN pre-trained models (animals and plants) and can process both individual sequences or fasta files. When making predictions on individual sequences, the server generates a nucleotide importance graph.

## Package installation

The latest version of the package can be downloaded from the GitHub [repository](https://github.com/cyones/mirDNN). The exact version used in the paper is allocated in [SourceForge](https://sourceforge.net/projects/sourcesinc/files/mirdnn).

To download from GitHub:

```bash
git clone --recurse-submodules https://github.com/cyones/mirDNN.git
```
After downloading the package (from GitHub or SourceForge), install the dependencies:

```bash
cd mirDNN
pip install -r requirements.txt
```

This install all the packages needed to run mirDNN. In order to train models or make predictions the secondary structure of the sequences has to be inferred. For this task, the [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) software should be use.

## Usage

To make predictions or training new models, the first step is to predict the secondary structure of the sequences to process. This can be done with RNAfold. For example, given a fasta file named *sequences.fa*, run:

```bash
RNAfold --noPS --infile=sequences.fa --outfile=sequences.fold
```

(an example of folded sequence [is provided](./sequences))

### Inference

Now that we have the *.fold* file, to make predictions with the [provided pre-trained model](./models) for animals, simply run:

```bash
python3 mirdnn_eval.py -i sequences/test.fold -o predictions.csv -m models/animal.pmt -s 160 -d "cpu"
```

To calculate nucleotide importance values the command is:

```bash
python3 mirdnn_explain.py -i sequences/test.fold -o importance.csv -m models/animal.pmt -s 160 -d "cpu"
```

### Training new models

To train new models, two *.fold* files are needed, one with negative examples (non pre-miRNA sequences) and other with positive examples (well-known pre-miRNAs).

Given these datasets, the training can be done with:

```bash
python3 mirdnn_fit.py -i negative_sequences.fold -i positive_sequences.fold -m out_model.pmt -l train.log -d "cuda:0" -s 160
```

> NOTE: training a model is a very computing intensive task, therefore, it is recommended to use a GPU.

For more details about the training parameters, execute

```bash
python3 mirddn_fit.py -h
```

## Reproduce experiments

To reproduce the experiments, [R](https://www.r-project.org/) must be installed.

All the experiments presented in the paper can be easily reproduced using the Makefile inside the folder [experiments](./experiments).  For example, to generate the PRROC curve obtained in *Caenorhabditis elegans*, run:

```bash
cd experiments
make results/PRROC-cel.pdf
```

You will be asked to download the sequences files, and then all the necessary commands to train and test the model will be automatically executed.
