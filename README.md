# Word Sense Disambiguation

This repository contains all code utilized for developing, training and testing the DeBERTa, RoBERTa and BERT-WWM models.

You will need to install the dependencies as follows:

```bash
$ poetry install
```

The datasets for training can be acquired from [here](http://lcl.uniroma1.it/wsdeval/training-data) and the evaluation datasets can be downloaded [here](http://lcl.uniroma1.it/wsdeval/evaluation-data).

It will be assumed that the datasets were downloaded into `train` and `test` respectively.


## Project Architecture
                                  
```bash
├── colour_logging.py
├── datagen
│   └── dataset.py
├── dataset_statistics.ipynb
├── main.py
├── modelling
│   ├── collator.py
│   ├── metrics.py
│   ├── model.py
│   └── trainer.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── saliency_lit.ipynb
├── saliency_own.ipynb
├── slurm
│   ├── datagen-eval-all.sh
│   ├── datagen-eval-deberta.sh
│   ├── datagen-eval-roberta.sh
│   ├── eval_checkpoints.sh
│   ├── model-eval-all.sh
│   ├── roberta-eval.sh
│   ├── semcor-datasets.sh
│   ├── semcor+omsti-datasets.sh
│   ├── semcor+omsti-training-probing.sh
│   ├── semcor+omsti-training.sh
│   ├── semcor-training-probing.sh
│   ├── semcor-training-raw.sh
│   └── semcor-training.sh
```

## Generating Datasets

This can be done by executing the `datagen` module with
```bash
$ poetry run python main.py datagen
```

Flag list:

```bash
Usage: main.py datagen [OPTIONS]

  transform datasets into a format compatible with the MLMs

Options:
  -ds, --dataset PATH             path to tokens and lemmata in XML
                                  [required]
  -gs, --gold-standard PATH       path to gold standard  [required]
  -hm, --hf-model [bert-wwm|roberta|deberta]
                                  supported huggingface models  [required]
  -op, --output-path PATH         path to output file  [required]
  -tr, --train-dataset PATH       Path to a training dataset from which to
                                  extract sense key ids (switches to test
                                  dataset generation mode)
  --help                          Show this message and exit.
```


## Training Dataset

```bash
$ poetry run python main.py datagen \
    -ds train/WSD_Training_Corpora/SemCor/semcor.data.xml \
    -gs train/WSD_Training_Corpora/SemCor/semcor.gold.key.txt \
    -hm roberta \
    -op dataset/roberta+semcor.pickle
```


Because the datasets generated also perform all necessary preprocessing by invoking each model's tokenizer, the `-hm` or `--hf-model` flag must be used to instantiate the correct tokenizer.
The command above will create two datasets in the `dataset` folder. 
The first one is  a file called `dataset/roberta+semcor.pickle` containing the merged data and gold standard files.
The second one is huggingface dataset, stored in a folder called `dataset/roberta+semcor.hf`.


## Test Dataset

```bash
poetry run python main.py datagen \
    -ds test/WSD_Unified_Evaluation_Datasets/semeval2013/semeval2013.data.xml \
    -gs test/WSD_Unified_Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt \ 
    -hm roberta \
    -op dataset/roberta+semeval2013.pickle \
    --train-dataset dataset/roberta+semcor.pickle
```

To generate a good test dataset, the `--train-dataset` is required to point to a training dataset, so that classification can be performed correctly.


# Modelling

This can be done by executing the `modelling` module with

```bash
$ poetry run python main.py modelling
```

```
Usage: main.py modelling [OPTIONS]

  train and test models

Options:
  model: [mutually_exclusive, required]
                                  select model to work with
    -hm, --hf-model [bert-wwm|roberta|deberta]
                                  supported huggingface models
    -lm, --local-model DIRECTORY  path to locally stored model
  workload: [mutually_exclusive, required]
                                  training or testing
    -tr, --train FILE             path to training set
    -te, --test FILE              path to test set
    -op, --output-path PATH       Where to store result  [required]
    -fm, --freeze-model           Freeze LM model parameters while training
    -e, --epoch INTEGER           how many epochs to train for
    --run-name TEXT               Name to be used to identify run in result
                                  dict
    --metric-file TEXT            File to write metrics
    --help                        Show this message and exit.
```

## Model Selection

Our script offers two ways of loading models; either the model is available from huggingface, or is locally stored at a known path.
For the former, we support loading BERT-WWM, RoBERTa and DeBERTa from the huggingface website using the `--hf-model` flag.
For the latter, locally stored (probed or fine-tuned) instances can be loaded using the `--local-model` flag.

## Training

Specify the `--train` flag to point at a training dataset generated using the previously mentioned steps for dataset generation.
You must specify a dataset created for the model referenced with one of the model flags (`--hf-model` or `--local-model`).
Despite passing only one filepath, both the huggingface dataset and the pickled dataset will be loaded.

Passing this flag prohibits you from passing the `--test` flag; this will have to be done in the next command.


Examples: 
1. Downloading RoBERTa, using the SemCor training dataset, probing for 10 epochs, and storing the result:
 
```bash
$ poetry run python main.py modelling \
    --hf-model roberta \
    --train dataset/roberta+semcor.pickle \
    --freeze-model \
    --epoch 10 \
    --output-path models/probed-roberta+semcor-e10
```

2. Loading a previously probed instance of RoBERTa, using the SemCor training dataset, fine-tuning the model for 5 epochs, and storing the result:

```bash
$ poetry run python main.py modelling \
    --local-model models/probed-roberta+semcor-e10 \
    --train dataset/roberta+semcor.pickle \
    --epoch 5 \
    --output-path models/finetuned-roberta+semcor-e5
```


## Testing 

Specify the `--test` flag to point at a testing dataset generated using the previously mentioned steps for dataset generation.
You must specify a dataset created for the model referenced with one of the model flags (`--hf-model` or `--local-model`).

3. Loading a finetuned instance of RoBERTa for testing upon the SemEval2007 evaluation corpus

```bash
$ poetry run python main.py modelling \
    --local-model models/finetuned-roberta+semcor-e5 \
    --test dataset/semeval2007.pickle \
    --output-path models/eval/finetuned-roberta+semcor-e5
    --run-name ft-roberta+semeval2007 \
    --metric-file out/eval/metrics.pickle
```