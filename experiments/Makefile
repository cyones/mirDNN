PYTORCH_DEV ?= "cuda:0" # "cpu", "cuda:1", etc.
VALID_PROP ?= 0.1
EARLY_STOP ?= 500

default:
	make all

clear:
	rm -f explain/*
	rm -f logfiles/*
	rm -f models/*
	rm -f predictions/*
	rm -f results/*
	$(MAKE) -C sequences clear

sequences/%:
	$(MAKE) -C sequences $*

models/ath.pmt logfiles/ath.out : \
	sequences/ath-train-pos.fold sequences/ath-train-neg.fold
	../mirdnn_fit.py \
		-i sequences/ath-train-neg.fold \
		-i sequences/ath-train-pos.fold \
		-m models/ath.pmt -d ${PYTORCH_DEV} -s 320 -l logfiles/ath.out \
		-e ${EARLY_STOP} -v ${VALID_PROP}
models/%.pmt logfiles/%.out : \
	sequences/%-train-pos.fold sequences/%-train-neg.fold
	../mirdnn_fit.py \
		-i sequences/$*-train-neg.fold \
		-i sequences/$*-train-pos.fold \
		-m models/$*.pmt -d ${PYTORCH_DEV} -s 160 -l logfiles/$*.out \
		-e ${EARLY_STOP} -v ${VALID_PROP}

predictions/mirDNN-ath-pos.csv predictions/mirDNN-ath-neg.csv : \
	sequences/ath-test-pos.fold sequences/ath-test-neg.fold models/ath.pmt
	../mirdnn_eval.py \
		-i sequences/ath-test-pos.fold \
		-i sequences/ath-test-neg.fold \
		-o predictions/mirDNN-ath-pos.csv \
		-o predictions/mirDNN-ath-neg.csv \
		-m models/ath.pmt -d $(PYTORCH_DEV) -s 320
predictions/mirDNN-%-pos.csv predictions/mirDNN-%-neg.csv : \
	sequences/%-test-pos.fold sequences/%-test-neg.fold models/%.pmt
	../mirdnn_eval.py \
		-i sequences/$*-test-pos.fold \
		-i sequences/$*-test-neg.fold \
		-o predictions/mirDNN-$*-pos.csv \
		-o predictions/mirDNN-$*-neg.csv \
		-m models/$*.pmt -d $(PYTORCH_DEV) -s 160

results/PRROC-%.pdf: \
	predictions/mirDNN-%-neg.csv predictions/mirDNN-%-pos.csv
	./utils/PRROC.py $*

results/barplot.pdf: \
	predictions/mirDNN-ath-neg.csv predictions/mirDNN-ath-pos.csv \
	predictions/mirDNN-cel-neg.csv predictions/mirDNN-cel-pos.csv \
	predictions/mirDNN-aga-neg.csv predictions/mirDNN-aga-pos.csv
	./utils/bar_plot.py

all: results/barplot.pdf results/PRROC-cel.pdf results/PRROC-aga.pdf \
     results/PRROC-ath.pdf results/PRROC-dre.pdf results/PRROC-hsa.pdf

.PHONY: all clear
.SECONDARY:
