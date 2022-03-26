#!/bin/bash

#SBATCH --job-name=semcor-training
#SBATCH --output=wsd_output.txt
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=students
#SBATCH --nodelist=gpu09

for dataset in senseval2 senseval3 semeval2007 semeval2013 semeval2015
do
	~/.local/bin/poetry run python main.py modelling \
		--local-model roberta+semcor \
       	--test dataset/roberta+${dataset}.pickle \
		--output-path eval/roberta+${dataset} \
done	
