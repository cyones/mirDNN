#!/usr/bin/Rscript --vanilla
args = commandArgs(trailingOnly=TRUE)
if(length(args) != 4)
	stop("4 arguments needed")

if(!require(seqinr, lib="/tmp")) {
	install.packages("seqinr",
			 lib="/tmp",
			 contrib="https://mirror.las.iastate.edu/CRAN/src/contrib")
	suppressPackageStartupMessages(library(seqinr, lib="/tmp"))
}

set.seed(as.integer(args[4]))
fss = read.fasta(args[3], as.string=T)
idx = sample(length(fss), round(length(fss) * as.numeric(args[2])))
if(args[1] == 'keep') {
	fss = fss[ idx]
} else {
	fss = fss[-idx]
}

write.fasta(fss, names=names(fss), file.out=args[3], as.string=T)

