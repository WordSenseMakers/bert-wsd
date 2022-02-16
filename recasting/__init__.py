import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

import pathlib

# from . import recast


@click.command(
    name="recasting",
    help="recast datasets to augment training and testing",
)
@click.option(
    "-di",
    "--dataset-in",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="path to locally stored dataset",
    required=True,
)
@click.option(
    "-do",
    "--dataset-out",
    type=click.Path(exists=False, readable=True, path_type=pathlib.Path),
    help="path to write recasted dataset to",
    required=True,
)
def main(**params):
    pass


if __name__ == "__main__":
    main()
