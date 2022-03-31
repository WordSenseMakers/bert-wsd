#!/bin/bash

#SBATCH --job-name=bertwwm-eval-datagen
#SBATCH --output=datagen-eval-bertwwm.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --partition=students
#SBATCH --mem=32GB


for dataset in senseval2 senseval3 semeval2007 semeval2013 semeval2015
do
  poetry run python main.py modelling \
    --dataset ./dataset/test/${dataset}/${dataset}.data.xml \
    --gold-standard ./dataset/test/${dataset}/${dataset}.gold.key.txt \
    --hf-model ${model} \
    --output-path ./dataset/${dataset}-${model}.pickle \
    --train-dataset dataset/semcor4roberta.pickle
done
