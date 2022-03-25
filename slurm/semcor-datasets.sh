#!/bin/bash

#SBATCH --job-name=semcor-datagen
#SBATCH --output=wsd_output.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --partition=students
#SBATCH --mem=32GB

for model in roberta deberta bert-wwm 
do
	~/.local/bin/poetry run python main.py datagen \
		-ds ./train/WSD_Training_Corpora/SemCor/semcor.data.xml \
		-gs ./train/WSD_Training_Corpora/SemCor/semcor.gold.key.txt \
		-hm ${model} \
		-op ./dataset/${model}+semcor.pickle
done
