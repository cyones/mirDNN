#!/usr/bin/Rscript --vanilla
args = commandArgs(trailingOnly=TRUE)
if(length(args) != 4)
	stop("4 arguments needed")

library(seqinr)


try(set.seed(as.integer(args[4])), silent=T)
fss = read.fasta(args[3], as.string=T)
nsamples = round(as.numeric(args[2]) * length(fss))
idx = sample(1:length(fss), nsamples, replace=F)
if(args[1] == "keep") {
	fss = fss[ idx]
} else {
	fss = fss[-idx]
}
write.fasta(fss, names=names(fss), file.out=args[3], as.string=T)

