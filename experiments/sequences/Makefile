NTHREADS ?= 8
TEST_SEED ?= 42
TRAIN_PROP ?= 0.95

clear :
	rm -f *-pos.fa *-neg.fa Genome-*.fa *.fold

%.fold : %.fa
	RNAfold --noPS --infile=$*.fa \
		--outfile=tmp.fold --jobs=${NTHREADS}
	mv tmp.fold $*.fold

Genome-%.tar.gz mB22-%.fa:
	printf "\n\n\nPlease download the files from https://sourceforge.net/projects/sourcesinc/files/mirdnn-data and extract them in the sequences folder\n\n\n"

%-train-neg.fa: Genome-%.tar.gz
	tar xvf Genome-$*.tar.gz
	./utils/sample_sequences.R keep ${TRAIN_PROP} Genome-$*.fa $(TEST_SEED)
	mv Genome-$*.fa $*-train-neg.fa

%-train-pos.fa:
	./utils/rm_specie.R $* mB22-no$*.fa $*-train-pos.fa

%-test-neg.fa: Genome-%.tar.gz
	tar xvf Genome-$*.tar.gz
	./utils/sample_sequences.R rm ${TRAIN_PROP} Genome-$*.fa $(TEST_SEED)
	./utils/blast_difference.R Genome-$*.fa mB22-no$*.fa
	mv Genome-$*.fa $*-test-neg.fa

%-test-pos.fa: mB22-no%.fa Genome-%.tar.gz
	./utils/grep_specie.R $* mB22-no$*.fa $*-test-pos.fa

.SECONDARY:
