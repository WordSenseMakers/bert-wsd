import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

import pathlib


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
        print(f"Fetching {params['hf_model']} from huggingface ...")
    else:
        print(f"Loading {params['local_model']} from storage ...")


if __name__ == "__main__":
    main()
