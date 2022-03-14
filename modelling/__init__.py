import pathlib

import numpy as np

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import datasets

import torch

import colour_logging as logging
from . import collators, metrics, trainer as trnr
from datagen.dataset import SemCorDataSet


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

    hf_model = params["hf_model"]
    if hf_model is not None:
        if hf_model == "bert-wwm":
            model_name = "bert-large-uncased-whole-word-masking"
        elif hf_model == "roberta":
            model_name = "roberta-base"
        else:
            assert hf_model == "deberta"
            model_name = "microsoft/deberta-base"
        logging.info(
            f"Fetching {params['hf_model']} ({model_name}) from huggingface ..."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    else:
        logging.info(f"Loading {params['local_model']} from storage ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)

    model = model.to(device)

    logging.success(f"Loaded {model_name}")
    out, tr_path, ts_path = params["output_path"], params["train"], params["test"]

    ds_path = tr_path or ts_path
    logging.info(f"Loading dataset from {ds_path}")
    ds = SemCorDataSet.unpickle(ds_path)
    logging.success(f"Loaded dataset")

    sentence_level = (
        ds.df.groupby(["docid", "sntid"])
        .agg({"token": " ".join})
        .rename(columns={"token": "sentence"})
    )
    if tr_path is not None:
        # sentence_level = sentence_level.sample(frac=1).reset_index(drop=True).head(n=n)
        logging.info(f"Tokenizing dataset and splitting into training and testing")
        tr_dataset = datasets.Dataset.from_pandas(sentence_level).map(
            lambda df: tokenizer(df["sentence"], padding="longest", truncation="longest_first"),
            batched=True
        ).select(range(10))#.shuffle()

        # For streaming
        # with tempfile.NamedTemporaryFile(dir=ds_path.parent) as trfile:
        #tmp_file = ds_path.parent / f"{ds_path.name}.tmp"
        #tr_dataset.save_to_disk(tmp_file)
        #streamed_dataset = datasets.IterableDataset(tr_dataset)
        #train_dataset = streamed_dataset.take(sentence_level.shape[0] // 0.8)
        #eval_dataset = streamed_dataset.take(sentence_level.shape[0] - sentence_level.shape[0] // 0.8)

        ds = tr_dataset.train_test_split(test_size=0.2)
        train_dataset = ds["train"]
        eval_dataset = ds["test"]
        logging.success("Successfully tokenized and split dataset")

        # metric = metrics.WordSenseSimilarity(dataset=ds, config_name="min")
        # dc = collators.DataCollatorForPreciseLanguageModeling(tokenizer=tokenizer, dataset=ds)
        tr_args = TrainingArguments(
            output_dir=out,
            evaluation_strategy="epoch",
            optim="adamw_torch",
            # remove_unused_columns=False
        )

        trainer = trnr.WordSenseTrainer(
            model=model,
            dataset=ds,
            tokenizer=tokenizer,
            args=tr_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            #compute_metrics=lambda ep: _compute_metrics(metric, ep),
            data_collator=DataCollatorForLanguageModeling(tokenizer),
        )
        trainer.train()
        trainer.save_model(out)
    
    elif ts_path is not None:
        trainer = Trainer(
            model=model,
            eval_dataset=dataset,
            #compute_metrics=lambda ep: _compute_metrics(metric, ep),
            data_collator=DataCollatorForLanguageModeling(tokenizer),
        )

        trainer.evaluate()

    else:
        raise AssertionError("Both training and testing were None!")

def _compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    wss = metric.compute(predictions=predictions, reference=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'wss': wss,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def _compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    main()
