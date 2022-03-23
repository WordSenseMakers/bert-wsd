import pathlib

import numpy as np, pandas as pd

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModel,
    DataCollatorForWholeWordMask,
)
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, load_metric

from nltk.corpus import wordnet as wn

import torch

import colour_logging as logging
from modelling.model import SynsetClassificationModel
from . import collators, metrics, trainer as trnr
from datagen.dataset import SemCorDataSet

import nltk

BERT_WHOLE_WORD_MASKING = "bert-large-uncased-whole-word-masking"


@click.command(name="modelling", help="train and test models")
@optgroup.group(
    name="model",
    help="select model to work with",
    cls=RequiredMutuallyExclusiveOptionGroup,
)
@optgroup.option(
    "-hm",
    "--hf-model",
    type=click.Choice(["bert-wwm", "roberta", "deberta"], case_sensitive=False),
    callback=lambda ctx, param, value: value.lower(),
    help="supported huggingface models",
)
@optgroup.option(
    "-lm",
    "--local-model",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
    help="path to locally stored model",
)
@optgroup.group(
    name="workload",
    help="training or testing",
    cls=RequiredMutuallyExclusiveOptionGroup,
)
@optgroup.option(
    "-tr",
    "--train",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
    help="path to training set",
)
@optgroup.option(
    "-te",
    "--test",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
    help="path to test set",
)
@click.option(
    "-op",
    "--output-path",
    type=click.Path(exists=False, readable=True, path_type=pathlib.Path),
    required=True,
    help="Where to store result",
)
def main(**params):
    if torch.cuda.is_available():
        device = "cuda:0"
        logging.info(f"CUDA found; running on {device}")
    else:
        device = "cpu"
        logging.info(f"CUDA not found; running on {device}")

    out, tr_path, ts_path = params["output_path"], params["train"], params["test"]

    ds_path = tr_path or ts_path
    logging.info(f"Loading dataset from {ds_path}")
    ds = SemCorDataSet.unpickle(ds_path.with_suffix(".pickle"))
    hf_ds = Dataset.load_from_disk(ds_path.with_suffix(".hf"))
    logging.success(f"Loaded dataset")

    hf_model = params["hf_model"]
    if hf_model is not None:
        model_name = construct_model_name(hf_model)
        logging.info(
            f"Fetching {params['hf_model']} ({model_name}) from huggingface ..."
        )
        logging.info("Loading classification model ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        cl_model = SynsetClassificationModel(model_name, ds.all_sense_keys.shape[0])
        logging.success("Loaded classification model")

    else:
        model_name = params["local_model"]
        logging.info(f"Loading {params['local_model']} from storage ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        cl_model = AutoModelForMaskedLM.from_pretrained(
            model_name, local_files_only=True
        )

    # ==> base_model_name [ + checkpoint ]
    cl_model = cl_model.to(device)
    logging.success(f"Loaded {model_name}")

    hf_ds = hf_ds.add_column("sense-labels", hf_ds["labels"])
    relevant_columns = [
        column
        for column in hf_ds.column_names
        if column not in ds.sentence_level.columns
    ]
    relevant_columns.append("sense-labels")
    hf_ds.set_format(type="torch", columns=relevant_columns)

    if tr_path is not None:
        # sentence_level = sentence_level.sample(frac=1).reset_index(drop=True).head(n=n)

        # For streaming
        # with tempfile.NamedTemporaryFile(dir=ds_path.parent) as trfile:
        # tmp_file = ds_path.parent / f"{ds_path.name}.tmp"
        # tr_dataset.save_to_disk(tmp_file)
        # streamed_dataset = IterableDataset(tr_dataset)
        # train_dataset = streamed_dataset.take(sentence_level.shape[0] // 0.8)
        # eval_dataset = streamed_dataset.take(sentence_level.shape[0] - sentence_level.shape[0] // 0.8)

        ds_splits = hf_ds.train_test_split(test_size=0.2, shuffle=False)
        train_dataset = ds_splits["train"]
        eval_dataset = ds_splits["test"]
        logging.success("Successfully tokenized and split dataset")
        # nltk.download('omw-1.4')

        # metric = metrics.WordSenseSimilarity(dataset=ds, config_name="min")
        # dc = collators.DataCollatorForPreciseLanguageModeling(tokenizer=tokenizer, dataset=ds)
        tr_args = TrainingArguments(
            output_dir=out,
            evaluation_strategy="epoch",
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        if model_name == BERT_WHOLE_WORD_MASKING:
            collator = DataCollatorForWholeWordMask(tokenizer)
        else:
            collator = DataCollatorForLanguageModeling(tokenizer)

        trainer = Trainer(
            model=cl_model,
            tokenizer=tokenizer,
            args=tr_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # compute_metrics=lambda ep: _compute_metrics(metric, ep),
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(out)

    elif ts_path is not None:

        metric = metrics.WordSenseSimilarity(dataset=ds)

        trainer = Trainer(
            model=cl_model,
            eval_dataset=ds,
            compute_metrics=lambda ep: _compute_metrics(tokenizer, ep),
            data_collator=DataCollatorForLanguageModeling(tokenizer),
        )

        eval_metrics = trainer.evaluate()

        for k, v in eval_metrics.items():
            print(f"{k}\t{v}")

    else:
        raise AssertionError("Both training and testing were None!")


def construct_model_name(hf_model):
    if hf_model == "bert-wwm":
        model_name = BERT_WHOLE_WORD_MASKING
    elif hf_model == "roberta":
        model_name = "roberta-base"
    else:
        assert hf_model == "deberta"
        model_name = "microsoft/deberta-base"
    return model_name


def _compute_metrics(tokenizer, eval_pred):
    logging.info(f"Fetching metrics from huggingface ...")
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    logging.success("Loaded metrics")

    # wss = metric.compute(predictions=predictions, reference=reference)

    logits, labels = eval_pred

    # Get IDs
    mask_mask = labels != -100
    predictions = np.argmax(logits, axis=-1)[mask_mask].flatten()
    reference = labels[mask_mask].flatten()

    # Set prediction = reference if there exists a synsets overlap
    def overlap(prediction: int, reference: int):
        syn1 = set(wn.synsets(tokenizer.decode(prediction).strip()))
        syn2 = set(wn.synsets(tokenizer.decode(reference).strip()))

        if len(syn1.intersection(syn2)) > 0:
            return reference

        return prediction

    predictions = list(map(overlap, predictions, reference))

    average = "weighted"

    return {
        # 'wss': wss,
        "accuracy": accuracy._compute(predictions, reference)["accuracy"],
        "precision": precision._compute(predictions, reference, average=average)[
            "precision"
        ],
        "recall": recall._compute(predictions, reference, average=average)["recall"],
        "f1_score": f1._compute(predictions, reference, average=average)["f1"],
    }


if __name__ == "__main__":
    main()
