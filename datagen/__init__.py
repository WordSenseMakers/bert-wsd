import pathlib, io

import click
import pandas as pd
import numpy as np
import colorama
from lxml import etree
from tqdm import tqdm
from transformers import AutoTokenizer

from modelling import construct_model_name
from .dataset import SemCorDataSet
from datasets import Dataset

import colour_logging as logging

from nltk.corpus import wordnet as wn

BERT_WHOLE_WORD_MASKING = "bert-large-uncased-whole-word-masking"

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
    "-hm",
    "--hf-model",
    type=click.Choice(["bert-wwm", "roberta", "deberta"], case_sensitive=False),
    callback=lambda ctx, param, value: value.lower(),
    help="supported huggingface models",
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
    model_name = params["hf_model"]
    hugging_ds, semcor_ds = _create_dataset(xf, gs, model_name)

    op = params["output_path"]
    hf_op = op.with_suffix(".hf")
    sc_op = op.with_suffix(".pickle")
    logging.success(f"Storing HuggingFace dataset in {hf_op}")
    hugging_ds.save_to_disk(hf_op)
    logging.success(f"Storing SemCor dataset in {sc_op}")
    semcor_ds.pickle(sc_op)



def _create_dataset(xmlfile: str, goldstandard: str, model_name: str):
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
    data_set = SemCorDataSet(df)

    pretrained_model_name = construct_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # todo: multi label classification
    def map_data(chunk: Dataset) -> dict:
        # map sense key for each token or falsy value (-100)
        tokenized = tokenizer(
            chunk["sentence"],
            padding="max_length",
            truncation="longest_first",
        )

        idxs = (
            chunk["sentence_idx"]
            if isinstance(chunk["sentence_idx"], list)
            else [chunk["sentence_idx"]]
        )
        tokens2senses = pd.merge(
            data_set.token_level,
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
                    # Only label the first token of a given word.
                    word_idx != previous_word_idx or pretrained_model_name == BERT_WHOLE_WORD_MASKING
                ):
                    r = tokens2senses[tokens2senses.sentence_idx == sentence_idx]
                    token_df = r[r.tokpos == word_idx]

                    if token_df["sense-keys"].isna().all():
                        # TODO: Figure out falsy value
                        label_ids.append(-100)
                    else:
                        row = token_df[["sense-key-idx1"]].fillna(-100, axis=1)
                        row = int(row.iloc[0])
                        label_ids.append(row)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    logging.info(f"Preprocessing HuggingFace dataset")
    hugging_dataset = (
        Dataset.from_pandas(data_set.sentence_level).map(map_data, batched=True)
    )

    stats = io.StringIO()
    df.info(buf=stats)
    logging.success(f"Merged!\n")
    logging.info(f"Statistics: {stats.getvalue()}")
    logging.info(f"{df.head()}")
    return hugging_dataset, data_set


if __name__ == "__main__":
    main()
