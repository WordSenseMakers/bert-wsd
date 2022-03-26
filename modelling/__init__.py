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
from . import metrics
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
    callback=lambda ctx, param, value: value.lower() if value is not None else None,
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
@click.option(
    "-fm",
    "--freeze-model",
    help="Freeze LM model parameters while training",
    is_flag=True,
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
        config = AutoConfig.from_pretrained(model_name)
        cl_model = SynsetClassificationModel(
            config,
            model_name,
            ds.all_sense_keys.shape[0],
            freeze_lm=params["freeze_model"],
        )
        logging.success("Loaded classification model")

    else:
        model_name = params["local_model"]
        base_model_name = construct_model_name(str(model_name))
        logging.info(f"Loading {params['local_model']} from storage ...")
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        cl_model = SynsetClassificationModel.from_pretrained(
            model_name,
            config=config,
            local_files_only=True,
            model_name=base_model_name,
            num_classes=2584,
            freeze_lm=params["freeze_model"],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    if model_name == BERT_WHOLE_WORD_MASKING:
        collator = BetterDataCollatorForWholeWordMask(tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer)

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

    metrics = load_metrics(ds)

    if tr_path is not None:
        ds_splits = hf_ds.train_test_split(
            test_size=0.2,
            #    shuffle=False,
        )
        train_dataset = ds_splits["train"]
        eval_dataset = ds_splits["test"]
        logging.success("Successfully split dataset")

        tr_args = TrainingArguments(
            output_dir=out,
            evaluation_strategy="steps",
            save_strategy="steps",
            optim="adamw_torch",
            remove_unused_columns=False,
            label_names=["labels", "sense-labels"],
            load_best_model_at_end=True,
            num_train_epochs=params["epoch"],
            fp16=True,
            eval_accumulation_steps=50,
            eval_steps=1,
        )

        trainer = BetterTrainer(
            model=cl_model,
            tokenizer=tokenizer,
            args=tr_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda ep: _compute_metrics(metrics, ep),
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(out)

    elif ts_path is not None:

        te_args = TrainingArguments(
            output_dir=out,
            remove_unused_columns=False,
            label_names=["labels", "sense-labels"],
        )

        trainer = BetterTrainer(
            model=cl_model,
            eval_dataset=hf_ds,
            compute_metrics=lambda ep: _compute_metrics(metrics, ep),
            data_collator=collator,
            args=te_args,
        )

        eval_metrics = trainer.evaluate()

        for k, v in eval_metrics.items():
            print(f"{k}\t{v}")

    else:
        raise AssertionError("Both training and testing were None!")


def construct_model_name(hf_model: str):
    if "bert-wwm" in hf_model:
        model_name = BERT_WHOLE_WORD_MASKING
    elif "roberta" in hf_model:
        model_name = "roberta-base"
    else:
        assert "deberta" in hf_model
        model_name = "microsoft/deberta-base"
    return model_name


def load_metrics(dataset: SemCorDataSet) -> list:
    logging.info(f"Loading huggingface metrics")
    average = "weighted"

    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    dataset = WordSenseSimilarity(dataset)
    f1 = load_metric("f1")

    computations = [
        lambda p, r: accuracy.compute(predictions=p, references=r),
        lambda p, r: precision.compute(predictions=p, references=r, average=average),
        lambda p, r: recall.compute(predictions=p, references=r, average=average),
        lambda p, r: dataset.compute(predictions=p, references=r),
        lambda p, r: f1.compute(predictions=p, references=r, average=average),
    ]
    logging.success("Loaded metrics")

    return computations


def _compute_metrics(metrics: list, eval_pred):
    logits, (masked_labels, sense_labels) = eval_pred
    labels = sense_labels[:]
    labels[masked_labels == -100] = -100

    # Get IDs
    lossable = labels != -100
    predictions = np.argmax(logits, axis=-1)[lossable.flatten()]
    reference = labels[lossable]

    result = dict()
    for metric in metrics:
        result.update(metric(predictions, reference))

    return result


if __name__ == "__main__":
    main()
