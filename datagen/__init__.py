import pathlib, io

import click
import pandas as pd
import numpy as np
import colorama

from .dataset import SemCorDataSet
import colour_logging as logging

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
    "-xp",
    "--xpath",
    type=str,
    help="xpath to individual sentences within given dataset",
    required=True
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
    type=click.Path(exists=False, writable=True, path_type=pathlib.Path),
    help="path to output file",
    required=True,
)
def main(**params):
    xf = params["dataset"]
    gs = params["gold_standard"]
    xp = params["xpath"] 
    ds = _create_dataset(xf, gs, xp)

    op = params["output_path"]
    logging.success(f"Storing dataset in {op}")
    ds.pickle(op)


def _create_dataset(xmlfile: str, goldstandard: str, xp: str) -> SemCorDataSet:
    logging.info(f"Loading tokens and lemmata from {xmlfile}")
    data_df = pd.read_xml(xmlfile, xpath=xp)
    data_df.rename(columns={"wf": "token"}, inplace=True)
    data_df.token.fillna(value=data_df.instance, inplace=True)

    data_df[["docid", "sntid", "tokid"]] = data_df.id.str.split(".", expand=True)
    data_df = data_df.drop(columns=["instance"])
    logging.success("Loaded tokens and lemmata!\n")


    logging.info(f"Loading sense keys from {goldstandard}")
    gold_df = pd.read_csv(
        goldstandard, sep=" ", names=["id", "sense-key1", "sense-key2", "sense-key3"]
    )
    gold_df["sense-keys"] = gold_df[["sense-key1", "sense-key2", "sense-key3"]].apply(
        lambda e: e.str.cat(sep=","), axis=1
    )
    gold_df = gold_df.drop(columns=["sense-key1", "sense-key2", "sense-key3"])
    logging.success(f"Loaded sense keys!\n")

    logging.info(f"Merging tokens and lemmata with sense keys")
    df = data_df.merge(gold_df, on="id", how="left")

    stats = io.StringIO()
    df.info(buf=stats)
    logging.success(f"Merged!\n")
    logging.info(f"Statistics: {stats.getvalue()}")
    return SemCorDataSet(df)


if __name__ == "__main__":
    main()
