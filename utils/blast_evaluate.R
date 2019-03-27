#!/usr/bin/Rscript --vanilla
args = commandArgs(trailingOnly=TRUE)

df = read.table(args[2])
correct = grepl(args[1], df[[1]])
TP = length(unique(df[correct, 2]))
FP = length(unique(setdiff(df[,2], unique(df[correct,2]))))
cat(paste(args[3], TP, FP, "\n", sep="\t"))
