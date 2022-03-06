import pathlib

import numpy as np

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from transformers import BertTokenizer, BertModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_metric

import colour_logging as logging
from . import metrics

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
    callback=str.lower,
    help="supported huggingface models",
)
@optgroup.option(
    "-lm",
    "--local-model",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, readable=True, path_type=pathlib.Path),
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
    type=click.Path(file_okay=True, dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
    help="path to training set",
)
@optgroup.option(
    "-te",
    "--test",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
    help="path to test set",
)
@click.option(
    "-op",
    "--output-path",
    type=click.Path(exists=False, readable=True, path_type=pathlib.Path),
)
def main(**params):
    hf_model = params["hf_model"]
    if hf_model is not None:
        if hf_model == "bert-wwm":
            model_name = "bert-large-uncased-whole-word-masking"
        elif hf_model == "roberta":
            model_name = "roberta-large"
        else:
            assert hf_model == "deberta"
            model_name = "roberta-large"
        logging.info(f"Fetching {params['hf_model']} ({model_name}) from huggingface ...")
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=False)
        model = BertModel.from_pretrained(model_name, local_files_only=False)

    else:
        logging.info(f"Loading {params['local_model']} from storage ...")
        tokenizer = BertTokenizer.from_pretrained(model, local_files_only=True)
        model = BertModel.from_pretrained(model, local_files_only=True)


    out = params["output_path"]
    if params["train"] is not None:
        ds = SemCorDataSet.unpickle(params["train"])
        metric = metrics.SynsetSimilarity(dataset=ds, config_name="max")
        tr_args = TrainingArguments(output_dir=out, evaluation_strategy="epoch")
        dc = DataCollatorForLanguageModeling(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=tr_args,
            compute_metrics=lambda ep: _compute_metrics(metric, ep),
            data_collator=dc
        )


def _compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    main()
