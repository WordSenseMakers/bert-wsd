#!/bin/bash

for dataset in senseval2 senseval3 semeval2007 semeval2013 semeval2015
do
	poetry run python main.py modelling \
		--dataset ./dataset/test/${dataset}/${dataset}.data.xml \
		-gs ./dataset/test/${dataset}/${dataset}.gold.key.txt \
		-hm deberta \
		-op ./dataset/deberta+${dataset}.pickle \
		-te
done