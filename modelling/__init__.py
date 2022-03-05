import pathlib

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from transformers import BertTokenizer, BertModel

import colour_logging as logging

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
    help="supported huggingface models",
)
@optgroup.option(
    "-lm",
    "--local-model",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="path to locally stored model",
)
@optgroup.group(
    name="action",
    help="select workload",
    cls=RequiredMutuallyExclusiveOptionGroup,
)
@optgroup.option(
    "-tr",
    "--train",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="path to training set",
)
@optgroup.option(
    "-te",
    "--test",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="path to test set",
)
def main(**params):
    if (hf_model := params["hf_model"]) is not None:
        if hf_model == "bert-wwm":
            model = "bert-large-uncased-whole-word-masking"
        elif hf_model == "roberta":
            model = "roberta-large"
        else:
            assert hf_model == "deberta"
            model = "roberta-large"
        logging.info(f"Fetching {params['hf_model']} ({model}) from huggingface ...")
        tokenizer = BertTokenizer.from_pretrained(model, local_files_only=False)
        
    else:
        logging.info(f"Loading {params['local_model']} from storage ...")
        tokenizer = BertTokenizer.from_pretrained(model, local_files_only=True)
        


if __name__ == "__main__":
    main()
