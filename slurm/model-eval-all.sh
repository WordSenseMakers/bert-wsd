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
    --local-model out/checkpoints/bert-wwm-probing+semcor/checkpoint-185900 \
    --test ./dataset/${dataset}-bert-wwm.pickle \
    --output-path ./out/eval/${dataset}-bert-wwm+probing \
    --run-name ${dataset}-bert-wwm+probing \
    --metric-file out/eval/metrics.pickle

  poetry run python main.py modelling \
    --local-model out/checkpoints/deberta-probing+semcor/checkpoint-185900 \
    --test ./dataset/${dataset}-deberta.pickle \
    --output-path ./out/eval/${dataset}-deberta+probing \
    --run-name ${dataset}-deberta+probing \
    --metric-file out/eval/metrics.pickle

  poetry run python main.py modelling \
    --local-model out/checkpoints/roberta-probing+semcor/checkpoint-185900 \
    --test ./dataset/${dataset}-roberta.pickle \
    --output-path ./out/eval/${dataset}-roberta+probing \
    --run-name ${dataset}-roberta+probing \
    --metric-file out/eval/metrics.pickle

  poetry run python main.py modelling \
    --local-model out/checkpoints/deberta-finetune+semcor-lr5e-5ep185900 \
    --test ./dataset/${dataset}-deberta.pickle \
    --output-path ./out/eval/${dataset}-deberta+finetune \
    --run-name ${dataset}-deberta+finetune \
    --metric-file out/eval/metrics.pickle
done