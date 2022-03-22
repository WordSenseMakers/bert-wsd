import pathlib, io

import click
import pandas as pd
import numpy as np
import colorama
from lxml import etree
from tqdm import tqdm

from .dataset import SemCorDataSet
import colour_logging as logging

from nltk.corpus import wordnet as wn



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
    type=click.Path(exists=False, writable=True, path_type=pathlib.Path),
    help="path to output file",
    required=True,
)
def main(**params):
    xf = params["dataset"]
    gs = params["gold_standard"]
    ds = _create_dataset(xf, gs)

    op = params["output_path"]
    logging.success(f"Storing dataset in {op}")
    ds.pickle(op)


def _create_dataset(xmlfile: str, goldstandard: str) -> SemCorDataSet:
    rows = list()

    logging.info(f"Loading tokens and lemmata from {xmlfile}")
    for event, elem in tqdm(
        etree.iterparse(xmlfile, events=("end",), tag="sentence")
    ):
        docid, sntid = elem.attrib["id"].split(".")
        for tok_pos, child in enumerate(elem):
            if child.tag in ("wf", "instance"):
                assert child.text is not None, f"{child}"
            if child.tag == "wf":
                rows.append(
                    (
                        np.nan,
                        docid,
                        sntid,
                        np.nan,
                        tok_pos,
                        child.text,
                        child.attrib["lemma"],
                    )
                )
            elif child.tag == "instance":
                _, _, tokid = child.attrib["id"].split(".")
                rows.append(
                    (
                        child.attrib["id"],
                        docid,
                        sntid,
                        tokid,
                        tok_pos,
                        child.text,
                        child.attrib["lemma"],
                    )
                )
            else:
                raise Exception(f"Unhandled tag in sentence: {child.tag}")
    logging.success("Loaded tokens and lemmata!\n")

    data_df = pd.DataFrame(
        rows, columns=["id", "docid", "sntid", "tokid", "tokpos", "token", "lemma"]
    )

    logging.info(f"Loading sense keys from {goldstandard}")
    gold_df = pd.read_csv(
        goldstandard, sep=" ", names=["id", "sense-key1", "sense-key2", "sense-key3"]
    )
    sense_keys = []
    for synset in wn.all_eng_synsets():
        for lemma in synset.lemmas():
            sense_keys.append(lemma.key())
    sense_keys = pd.DataFrame(list(dict.fromkeys(sense_keys)), columns=["sense-keys"])
    sense_keys["sense-key-idx"] = pd.factorize(sense_keys["sense-keys"])[0]

    gold_df["sense-key-idx1"] = pd.merge(gold_df, sense_keys, how='left', left_on="sense-key1", right_on="sense-keys")["sense-key-idx"]
    gold_df["sense-key-idx2"] = pd.merge(gold_df, sense_keys, how='left', left_on="sense-key2", right_on="sense-keys")["sense-key-idx"]
    gold_df["sense-key-idx3"] = pd.merge(gold_df, sense_keys, how='left', left_on="sense-key3", right_on="sense-keys")["sense-key-idx"]

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
    logging.info(f"{df.head()}")
    return SemCorDataSet(df)


if __name__ == "__main__":
    main()
