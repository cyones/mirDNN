PYTORCH_DEV ?= "cuda:0" # "cpu", "cuda:1", etc.
SERVER ?= pc48.local
NTHREADS ?= 8
VALID_SEED ?= NA
TEST_SEED ?= 42
TRAIN_PROP ?= 0.9
VALID_PROP ?= 0.1
EARLY_STOP ?= 200


define fold
	RNAfold --noPS --infile=${1} --outfile=tmpRNAfold --jobs=${NTHREADS}
	mv tmpRNAfold ${2}
	rm ${1}
endef

default:
	rsync -atP . ${SERVER}:~/mirDNN/ --exclude '__pycache__' --exclude '*.swp'
	ssh -t ${SERVER} 'cd mirDNN; make clear; make run'
	rsync -atP ${SERVER}:~/mirDNN/results/ ./results
	touch ./results/*

clear:
	rm -f predictions/*
	rm -f logfiles/*
	rm -f results/*
	rm -f models/*

clear_all: clear
	rm -f datasets/*

datasets/%-train-neg.fold: sequences/Genome-%.tar.gz
	tar xvf sequences/Genome-$*.tar.gz
	./utils/sample_sequences.R keep ${TRAIN_PROP} Genome-$*.fa $(TEST_SEED)
	$(call fold,Genome-$*.fa,datasets/$*-train-neg.fold)

datasets/ath-train-pos.fold: sequences/mB22-viridiplantae.fa
	$(eval TMPFILE1 := $(shell mktemp))
	$(eval TMPFILE2 := $(shell mktemp))
	./utils/grep_specie.R ath sequences/mB22-viridiplantae.fa ${TMPFILE1}
	cp sequences/mB22-viridiplantae.fa ${TMPFILE2}
	./utils/blast_difference.R ${TMPFILE2} $(TMPFILE1)
	$(call fold,$(TMPFILE2),datasets/ath-train-pos.fold)
	rm $(TMPFILE1)
datasets/%-train-pos.fold: sequences/mB22-metazoa.fa
	$(eval TMPFILE1 := $(shell mktemp))
	$(eval TMPFILE2 := $(shell mktemp))
	./utils/grep_specie.R $* sequences/mB22-metazoa.fa ${TMPFILE1}
	cp sequences/mB22-metazoa.fa ${TMPFILE2}
	./utils/blast_difference.R ${TMPFILE2} $(TMPFILE1)
	$(call fold,$(TMPFILE2),datasets/$*-train-pos.fold)
	rm $(TMPFILE1)

datasets/ath-test-neg.fold: sequences/Genome-ath.tar.gz
	tar xvf sequences/Genome-ath.tar.gz
	./utils/sample_sequences.R rm ${TRAIN_PROP} Genome-ath.fa $(TEST_SEED)
	./utils/blast_difference.R Genome-ath.fa sequences/mB22-viridiplantae.fa
	$(call fold,Genome-ath.fa,datasets/ath-test-neg.fold)
datasets/%-test-neg.fold: sequences/Genome-%.tar.gz
	tar xvf sequences/Genome-$*.tar.gz
	./utils/sample_sequences.R rm ${TRAIN_PROP} Genome-$*.fa $(TEST_SEED)
	./utils/blast_difference.R Genome-$*.fa sequences/mB22-metazoa.fa
	$(call fold,Genome-$*.fa,datasets/$*-test-neg.fold)

datasets/ath-test-pos.fold: sequences/mB22-viridiplantae.fa sequences/Genome-ath.tar.gz
	$(eval TMPFILE := $(shell mktemp))
	./utils/grep_specie.R ath sequences/mB22-viridiplantae.fa ${TMPFILE}
	$(call fold,${TMPFILE},datasets/ath-test-pos.fold)
datasets/%-test-pos.fold: sequences/mB22-metazoa.fa sequences/Genome-%.tar.gz
	$(eval TMPFILE := $(shell mktemp))
	./utils/grep_specie.R $* sequences/mB22-metazoa.fa ${TMPFILE}
	$(call fold,${TMPFILE},datasets/$*-test-pos.fold)

logfiles/ath-optimization.out : \
	datasets/ath-train-pos.fold datasets/ath-train-neg.fold
	python3.5 model_optimize.py \
		-i datasets/ath-train-neg.fold -b 4096 \
		-i datasets/ath-train-pos.fold -b 2048 \
		-d ${PYTORCH_DEV} -s 320 -L logfiles/ath-optimization.out \
		-e ${EARLY_STOP} -v ${VALID_PROP} -r ${VALID_SEED}
logfiles/%-optimization.out : \
	datasets/%-train-pos.fold datasets/%-train-neg.fold
	python3.5 model_optimize.py \
		-i datasets/$*-train-neg.fold -b 8192 \
		-i datasets/$*-train-pos.fold -b 4096 \
		-d ${PYTORCH_DEV} -s 160 -L logfiles/$*-optimization.out \
		-e ${EARLY_STOP} -v ${VALID_PROP} -r ${VALID_SEED}

models/ath.pmt logfiles/ath.out : \
	datasets/ath-train-pos.fold datasets/ath-train-neg.fold
	python3.5 model_fit.py \
		-i datasets/ath-train-neg.fold -b 4096 \
		-i datasets/ath-train-pos.fold -b 2048 \
		-m models/ath.pmt -d ${PYTORCH_DEV} -s 320 -L logfiles/ath.out \
		-e ${EARLY_STOP} -v ${VALID_PROP} -r ${VALID_SEED}
models/%.pmt logfiles/%.out : \
	datasets/%-train-pos.fold datasets/%-train-neg.fold
	python3.5 model_fit.py \
		-i datasets/$*-train-neg.fold -b 8192 \
		-i datasets/$*-train-pos.fold -b 4096 \
		-m models/$*.pmt -d ${PYTORCH_DEV} -s 160 -L logfiles/$*.out \
		-e ${EARLY_STOP} -v ${VALID_PROP} -r ${VALID_SEED}

predictions/ath-test-pos.csv predictions/ath-test-neg.csv : \
	datasets/ath-test-pos.fold datasets/ath-test-neg.fold models/ath.pmt
	python3.5 model_eval.py \
		-i datasets/ath-test-pos.fold \
		-i datasets/ath-test-neg.fold \
		-o predictions/ath-test-pos.csv \
		-o predictions/ath-test-neg.csv \
		-m models/ath.pmt -d $(PYTORCH_DEV) -b 4096 -s 320
predictions/%-test-pos.csv predictions/%-test-neg.csv : \
	datasets/%-test-pos.fold datasets/%-test-neg.fold models/%.pmt
	python3.5 model_eval.py \
		-i datasets/$*-test-pos.fold \
		-i datasets/$*-test-neg.fold \
		-o predictions/$*-test-pos.csv \
		-o predictions/$*-test-neg.csv \
		-m models/$*.pmt -d $(PYTORCH_DEV) -b 8192 -s 160

results/%-loss.png results/%-AUC.csv : logfiles/%.out
	./utils/process_training.R logfiles/$*.out \
				   results/$*-loss.png \
				   results/$*-best.csv

results/%-ROC.png results/%-AUC.csv : predictions/%-test-pos.csv predictions/%-test-neg.csv
	./utils/process_test.R predictions/$*-test-pos.csv \
			       predictions/$*-test-neg.csv \
			       results/$*-ROC.png \
			       results/$*-AUC.csv

ROCs_small: results/ath-ROC.png results/cel-ROC.png results/aga-ROC.png
ROCs: ROCs_small results/dre-ROC.png results/hsa-ROC.png
losses_small: results/ath-loss.png results/cel-loss.png results/aga-loss.png
losses: losses_small results/dre-loss.png results/hsa-loss.png
all_small: ROCs_small losses_small
all: ROCs losses
run: all_small
.PHONY: run ROCs_small ROCs losses_small losses all all_small clear clear_all
