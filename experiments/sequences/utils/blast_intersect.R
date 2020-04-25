#!/usr/bin/Rscript --vanilla
args = commandArgs(trailingOnly=TRUE)
if(length(args) < 2)
	stop("2 arguments needed")

if(!require(seqinr)) {
	install.packages("seqinr", lib=.libPaths()[1],
			 contrib="https://mirror.las.iastate.edu/CRAN/src/contrib")
	suppressPackageStartupMessages(library(seqinr))
}

aux = tempfile()
ret = system(paste('./utils/qfblast', args[1], args[2], aux))
fs = as.character(unique(read.table(aux)[[1]]))
file.remove(aux)
ss = read.fasta(args[1], as.string=T)
ss = ss[fs]
if(length(args) < 3) {
	write.fasta(ss, names=names(ss), file.out=args[1], as.string=T)
} else {
	write.fasta(ss, names=names(ss), file.out=args[3], as.string=T)
}

