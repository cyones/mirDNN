PYTORCH_DEV ?= "cuda:0" # "cpu", "cuda:1", etc.
NTHREADS ?= 8
VALID_SEED ?= NA
VALID_PROP ?= 0.1
EARLY_STOP ?= 100

default:
	make all

clear:
	rm -f predictions/*
	rm -f logfiles/*
	rm -f results/*
	rm -f models/*

clear_all: clear
	rm -f sequences/*.fold

sequences/%:
	$(MAKE) -C sequences $*

tunning/ath.csv logfiles/ath_params.out : \
	sequences/ath-train-pos.fold sequences/ath-train-neg.fold
	python3.6 model_tune.py \
		-i sequences/ath-train-neg.fold \
		-i sequences/ath-train-pos.fold \
		-o tunning/ath.csv -d ${PYTORCH_DEV} -s 320 \
		-l logfiles/ath_tunning.out -e ${EARLY_STOP} \
		-v ${VALID_PROP} -r ${VALID_SEED}
tunning/%.csv logfiles/%_params.out : \
	sequences/%-train-pos.fold sequences/%-train-neg.fold
	python3.6 model_tune.py \
		-i sequences/$*-train-neg.fold \
		-i sequences/$*-train-pos.fold \
		-o tunning/$*.csv -d ${PYTORCH_DEV} -s 160 \
		-l logfiles/$*_tunning.out -e ${EARLY_STOP} \
		-v ${VALID_PROP} -r ${VALID_SEED}

models/ath.pmt logfiles/ath.out : \
	sequences/ath-train-pos.fold sequences/ath-train-neg.fold
	python3.6 model_fit.py \
		-i sequences/ath-train-neg.fold \
		-i sequences/ath-train-pos.fold \
		-m models/ath.pmt -d ${PYTORCH_DEV} -s 320 -l logfiles/ath.out \
		-e ${EARLY_STOP} -v ${VALID_PROP} -r ${VALID_SEED}
models/%.pmt logfiles/%.out : \
	sequences/%-train-pos.fold sequences/%-train-neg.fold
	python3.6 model_fit.py \
		-i sequences/$*-train-neg.fold \
		-i sequences/$*-train-pos.fold \
		-m models/$*.pmt -d ${PYTORCH_DEV} -s 160 -l logfiles/$*.out \
		-e ${EARLY_STOP} -v ${VALID_PROP} -r ${VALID_SEED}

predictions/mirDNN-ath-pos.csv predictions/mirDNN-ath-neg.csv : \
	sequences/ath-test-pos.fold sequences/ath-test-neg.fold models/ath.pmt
	python3.6 model_eval.py \
		-i sequences/ath-test-pos.fold \
		-i sequences/ath-test-neg.fold \
		-o predictions/mirDNN-ath-pos.csv \
		-o predictions/mirDNN-ath-neg.csv \
		-m models/ath.pmt -d $(PYTORCH_DEV) -s 320
predictions/mirDNN-%-pos.csv predictions/mirDNN-%-neg.csv : \
	sequences/%-test-pos.fold sequences/%-test-neg.fold models/%.pmt
	python3.6 model_eval.py \
		-i sequences/$*-test-pos.fold \
		-i sequences/$*-test-neg.fold \
		-o predictions/mirDNN-$*-pos.csv \
		-o predictions/mirDNN-$*-neg.csv \
		-m models/$*.pmt -d $(PYTORCH_DEV) -s 160

predictions/BLAST-%-pos.csv predictions/BLAST-%-neg.csv : \
	sequences/%-train-pos.fa sequences/%-test-pos.fa sequences/%-test-neg.fa
	./other/blast_classify.R   sequences/$*-train-pos.fa \
				   sequences/$*-test-pos.fa \
				   sequences/$*-test-neg.fa \
				   predictions/BLAST-$*-pos.csv \
				   predictions/BLAST-$*-neg.csv
predictions/RF-%-pos.csv predictions/RF-%-neg.csv : features/%-train-pos.csv
	./other/RF_classify.R $* predictions/RF-$*-pos.csv \
				 predictions/RF-$*-neg.csv
predictions/GBM-%-pos.csv predictions/GBM-%-neg.csv : features/%-train-pos.csv
	./other/GBM_classify.R $* predictions/GBM-$*-pos.csv \
				 predictions/GBM-$*-neg.csv
predictions/miRNAss-%-pos.csv predictions/miRNAss-%-neg.csv : features/%-train-pos.csv
	./other/miRNAss_classify.R $* predictions/miRNAss-$*-pos.csv \
				      predictions/miRNAss-$*-neg.csv

results/barplot.pdf:
	./utils/bar_plot.py
results/PRROC-%.pdf:
	./utils/PRROC.py $*

all: results/barplot.pdf results/PRROC-cel.pdf results/PRROC-aga.pdf \
     results/PRROC-ath.pdf results/PRROC-dre.pdf results/PRROC-hsa.pdf

.PHONY: all clear clear_all
.SECONDARY:
