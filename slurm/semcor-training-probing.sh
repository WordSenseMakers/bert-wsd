#!/bin/bash

#SBATCH --job-name=semcor-training-probing
#SBATCH --output=wsd_output.txt
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=students
#SBATCH --nodelist=gpu09

for model in roberta deberta bert-wwm 
do
	~/.local/bin/poetry run python main.py modelling \
		--hf-model ${model} \
       	--train dataset/${model}+semcor.pickle \
		--output-path models/${model}-probing+semcor \
		--epoch 10 \
    --freeze-model
done	
