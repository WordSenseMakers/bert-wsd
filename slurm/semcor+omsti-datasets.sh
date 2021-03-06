#!/bin/bash

#SBATCH --job-name=semcor+omsti-datagen
#SBATCH --output=datagen.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --partition=students
#SBATCH --mem=64GB

for model in roberta deberta bert-wwm 
do
	~/.local/bin/poetry run python main.py datagen \
		-ds ./train/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml \
		-gs ./train/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt \
		-hm ${model} \
		-op ./dataset/${model}+semcor+omsti.pickle
done