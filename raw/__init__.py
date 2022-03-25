import pathlib

import numpy as np, pandas as pd

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from modelling.collator import BetterDataCollatorForWholeWordMask
from modelling.metrics import WordSenseSimilarity

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModel,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizer,
    AutoConfig,
)
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, load_metric

from nltk.corpus import wordnet as wn

import torch

import colour_logging as logging
from modelling.model import SynsetClassificationModel
from modelling.trainer import BetterTrainer
from modelling import metrics
from datagen.dataset import SemCorDataSet

import nltk

BERT_WHOLE_WORD_MASKING = "bert-large-uncased-whole-word-masking"


@click.command(name="raw", help="motivate need for training classification model")
@click.option(
    "-hm",
    "--hf-model",
    type=click.Choice(["bert-wwm", "roberta", "deberta"], case_sensitive=False),
    callback=lambda ctx, param, value: value.lower() if value is not None else None,
    help="supported huggingface models",
    required=True
)
@click.option(
    "-ds",
    "--dataset",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
    help="path to data set",
    required=True,
)
@click.option(
    "-op",
    "--output-path",
    type=click.Path(exists=False, readable=True, path_type=pathlib.Path),
    required=True,
    help="Where to store result",
)
@click.option(
    "-e",
    "--epoch",
    help="how many epochs to train for",
    required=False,
    type=int,
    default=10,
)
def main(**params):
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    if torch.cuda.is_available():
        device = "cuda:0"
        logging.info(f"CUDA found; running on {device}")
    else:
        device = "cpu"
        logging.info(f"CUDA not found; running on {device}")

    out, ds_path = params["output_path"], params["dataset"]

    logging.info(f"Loading dataset from {ds_path}")
    ds = SemCorDataSet.unpickle(ds_path.with_suffix(".pickle"))
    hf_ds = Dataset.load_from_disk(ds_path.with_suffix(".hf"))
    logging.success(f"Loaded dataset")

    hf_model = params["hf_model"]

    model_name = construct_model_name(hf_model)
    logging.info(
        f"Fetching {params['hf_model']} ({model_name}) from huggingface ..."
    )
    logging.info("Loading classification model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    logging.success("Loaded classification model")

    if model_name == BERT_WHOLE_WORD_MASKING:
        collator = BetterDataCollatorForWholeWordMask(tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer)

    model = model.to(device)
    logging.success(f"Loaded {model_name}")

    hf_ds = hf_ds.add_column("sense-labels", hf_ds["labels"])
    relevant_columns = [
        column
        for column in hf_ds.column_names
        if column not in ds.sentence_level.columns
    ]
    relevant_columns.append("sense-labels")
    hf_ds.set_format(type="torch", columns=relevant_columns)

    metrics = load_metrics(ds)

    te_args = TrainingArguments(
        output_dir=out,
        remove_unused_columns=False,
        label_names=["labels", "sense-labels"],
    )

    trainer = BetterTrainer(
        model=model,
        eval_dataset=hf_ds,
        compute_metrics=lambda ep: _compute_metrics(metrics, ep),
        data_collator=collator,
        args=te_args,
    )

    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        print(f"{k}\t{v}")

if __name__ == "__main__":
    main()
