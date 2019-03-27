#!/usr/bin/Rscript --vanilla
suppressPackageStartupMessages(library(tidyverse))
args = commandArgs(trailingOnly=TRUE)


df = read.table(fl, sep='\t', header=T)

g = df %>% select(trainL, nbatch, validL) %>%
	gather(loss, value, -nbatch) %>%
	ggplot(aes(y=value, x=nbatch, color=loss)) + geom_point(size=0.1)
png(args[2], width=800, height=600)
print(g)
dummy = dev.off()


df = data.frame(bestBatch = which.min(df$validLoss), validLoss = min(df$validLoss))
write.table(df, file = args[3], quote=F, row.names=F)
