import pathlib

import click
import pandas as pd
import numpy as np
import colorama

from .dataset import SemCorDataSet


@click.command(
    name="datagen", help="transform datasets into a format compatible with the MLMs"
)
@click.option(
    "-ds",
    "--dataset",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="path to tokens and lemmata in XML",
    required=True,
)
@click.option(
    "-gs",
    "--gold-standard",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="path to gold standard",
    required=True,
)
@click.option(
    "-op",
    "--output-path",
    type=click.Path(writable=True, path_type=pathlib.Path),
    help="path to output file",
    required=True,
)
def main(**params):
    xf = params["dataset"]
    gs = params["gold_standard"] 
    ds = _create_dataset(xf, gs)

    op = params["output_path"]
    print(f"{colorama.Fore.GREEN}Storing dataset in {op}")
    ds.pickle(op)


def _create_dataset(xmlfile: str, goldstandard: str) -> SemCorDataSet:
    print(f"{colorama.Fore.YELLOW}Loading tokens and lemmata from {xmlfile}")
    data_df = pd.read_xml(xmlfile, xpath="./text/sentence/*")
    data_df.rename(columns={"wf": "token"}, inplace=True)
    data_df.token.fillna(value=data_df.instance, inplace=True)

    data_df[["docid", "sntid", "tokid"]] = data_df.id.str.split(".", expand=True)
    data_df = data_df.drop(columns=["instance"])
    print(f"{colorama.Fore.GREEN}Successfully loaded tokens and lemmata!\n")


    print(f"{colorama.Fore.YELLOW}Loading sense keys from {goldstandard}")
    gold_df = pd.read_csv(
        goldstandard, sep=" ", names=["id", "sense-key1", "sense-key2", "sense-key3"]
    )
    gold_df["sense-keys"] = gold_df[["sense-key1", "sense-key2", "sense-key3"]].apply(
        lambda e: e.str.cat(sep=","), axis=1
    )
    gold_df = gold_df.drop(columns=["sense-key1", "sense-key2", "sense-key3"])
    print(f"{colorama.Fore.GREEN}Successfully sense keys!\n")

    print(f"{colorama.Fore.YELLOW}Merging tokens and lemmata with sense keys")
    df = data_df.merge(gold_df, on="id", how="left")
    print(f"{colorama.Fore.GREEN}Successfully merged!\n")
    return SemCorDataSet(df)


if __name__ == "__main__":
    main()
