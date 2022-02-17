import pathlib

import click
import pandas as pd
import numpy as np


@click.command(
    name="datagen", help="transform datasets into a format compatible with the MLMs"
)
@click.option(
    "-xf",
    "--xml-file",
    type=click.Path(exists=True, readable=True, path_type=pathlib.Path),
    help="XML file containing training data",
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
    xf = params["xml_file"]
    print("Loaded sentences")
    sentence_df = pd.read_xml(xf)
    sentence_df.fillna(value=np.nan, inplace=True)

    df = pd.read_xml(xf, xpath="./text/sentence/*")
    df.fillna(value=np.nan, inplace=True)
    print("Loaded instances and wfs")
    df = df.rename(columns={"id": "instance_id"})
    df = df.assign(text_id=np.nan, sentence_id=np.nan)

    print(df.loc[len(df["instance_id"].str.split(".")) == 3])

    print("Splitting ids")
    df[["text_id", "sentence_id", "instance_id"]] = df[
        pd.notna(df.instance_id)
    ].instance_id.str.split(".")
    print(df)


if __name__ == "__main__":
    main()
