import pathlib

import numpy as np, pandas as pd

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
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
    ds = SemCorDataSet.unpickle(ds_path)
    logging.success(f"Loaded dataset")

    hf_model = params["hf_model"]
    if hf_model is not None:
        if hf_model == "bert":
            model_name = "bert-large-uncased-whole-word-masking"
        elif hf_model == "roberta":
            model_name = "roberta-base"
        else:
            assert hf_model == "deberta"
            model_name = "microsoft/deberta-base"
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

    def map_data(chunk: Dataset) -> dict:
        # map sense key for each token or falsy value (-100)
        tokenized = tokenizer(
            chunk["sentence"],
            padding="longest",
            truncation="longest_first",
        )

        idxs = (
            chunk["sentence_idx"]
            if isinstance(chunk["sentence_idx"], list)
            else [chunk["sentence_idx"]]
        )
        tokens2senses = pd.merge(
            ds.token_level,
            pd.DataFrame(idxs, columns=["sentence_idx"]),
            how="inner",
            on="sentence_idx",
        )

        labels = []
        for i, sentence_idx in enumerate(idxs):
            # Map tokens to their respective word.
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    r = tokens2senses[tokens2senses.sentence_idx == sentence_idx]
                    token_df = r[r.tokpos == word_idx]

                    if token_df["sense-keys"].isna().all():
                        # TODO: Figure out falsy value
                        label_ids.append(None)
                    else:
                        label_ids.append(int(token_df["sense-key-idx1"].iloc[0]))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    logging.info(f"Preprocessing dataset and splitting into training and testing")
    dataset = (
        Dataset.from_pandas(ds.sentence_level).map(map_data, batched=True).shuffle()
    )
    print(dataset)

    relevant_columns = [
        column
        for column in dataset.column_names
        if column not in ds.sentence_level.columns
    ]
    relevant_columns.append("sentence_idx")
    dataset.set_format(type="torch", columns=relevant_columns)

    if tr_path is not None:
        # sentence_level = sentence_level.sample(frac=1).reset_index(drop=True).head(n=n)

        # For streaming
        # with tempfile.NamedTemporaryFile(dir=ds_path.parent) as trfile:
        # tmp_file = ds_path.parent / f"{ds_path.name}.tmp"
        # tr_dataset.save_to_disk(tmp_file)
        # streamed_dataset = IterableDataset(tr_dataset)
        # train_dataset = streamed_dataset.take(sentence_level.shape[0] // 0.8)
        # eval_dataset = streamed_dataset.take(sentence_level.shape[0] - sentence_level.shape[0] // 0.8)

        ds_splits = dataset.train_test_split(test_size=0.2)
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

        trainer = trnr.WordSenseTrainer(
            model=model,
            dataset=ds,
            tokenizer=tokenizer,
            args=tr_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # compute_metrics=lambda ep: _compute_metrics(metric, ep),
            data_collator=DataCollatorForLanguageModeling(tokenizer),
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
        #'wss': wss,
        "accuracy": accuracy._compute(predictions, reference)["accuracy"],
        "precision": precision._compute(predictions, reference, average=average)[
            "precision"
        ],
        "recall": recall._compute(predictions, reference, average=average)["recall"],
        "f1_score": f1._compute(predictions, reference, average=average)["f1"],
    }


if __name__ == "__main__":
    main()
