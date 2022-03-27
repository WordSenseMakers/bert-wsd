#!/bin/bash

#SBATCH --job-name=bertwwm-eval-datagen
#SBATCH --output=datagen-eval-bertwwm.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --partition=students
#SBATCH --mem=32GB

for dataset in senseval2 senseval3 semeval2007 semeval2013 semeval2015
do
	~/.local/bin/poetry run python main.py datagen \
		-ds ./test/${dataset}.data.xml \
		-gs ./test/${dataset}.gold.key.txt \
		-hm bert-wwm \
		-op ./dataset/bert-wwm+${dataset}.pickle \
		-te
done
