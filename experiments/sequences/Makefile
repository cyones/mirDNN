NTHREADS ?= 8
TEST_SEED ?= 42
TRAIN_PROP ?= 0.95

%.fold : %.fa
	RNAfold --noPS --infile=$*.fa \
		--outfile=tmp.fold --jobs=${NTHREADS}
	mv tmp.fold $*.fold

%-train-neg.fa: Genome-%.tar.gz
	tar xvf Genome-$*.tar.gz
	./utils/sample_sequences.R keep ${TRAIN_PROP} Genome-$*.fa $(TEST_SEED)
	mv Genome-$*.fa $*-train-neg.fa

%-train-pos.fa:
	./utils/rm_specie.R $* mB22-$*.fa $*-train-pos.fa

%-test-neg.fa: Genome-%.tar.gz
	tar xvf Genome-$*.tar.gz
	./utils/sample_sequences.R rm ${TRAIN_PROP} Genome-$*.fa $(TEST_SEED)
	./utils/blast_difference.R Genome-$*.fa mB22-$*.fa
	mv Genome-$*.fa $*-test-neg.fa

%-test-pos.fa: mB22-%.fa Genome-%.tar.gz
	./utils/grep_specie.R $* mB22-$*.fa $*-test-pos.fa

.SECONDARY:
