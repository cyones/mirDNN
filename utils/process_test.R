#!/usr/bin/Rscript --vanilla
suppressPackageStartupMessages(library(ROCR))
args = commandArgs(trailingOnly=TRUE)

X1 = read.table(args[1], header=F, sep=',')
X1$class = T
X2 = read.table(args[2], header=F, sep=',')
X2$class = F
X = rbind(X1, X2)
X$pred = X$V3

pred = prediction(X$pred, X$class)
ROC = performance(pred, "tpr", "fpr")
AUC = performance(pred, "auc")@y.values[[1]]
cat("AUC: ", AUC, "\n")
png(args[3], width=1200, height=900)
plot(ROC)
title(main=paste0(' AUC: ', AUC))
dummy = dev.off()

write.table(AUC, file=args[4], row.names=F, col.names = F)

