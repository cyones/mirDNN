#!/bin/zsh

makeblastdb -in $2 -dbtype nucl > /dev/null

touch .results/blast.csv

sp="ath"
for ev in 5 10 15 20 25 30 35 40 45 50
do
	DIR=`mktemp`
	blastn -query $1 -db $2 -task blastn -outfmt 6 -strand plus -out ${DIR} -num_threads 4 -evalue 1e-${ev} -dust no
	./utils/blast_evaluate.R ${sp} ${DIR} ${ev} >> ./results/blast.csv
done
rm $2.*nsq $2.*nin $2.*nhr
