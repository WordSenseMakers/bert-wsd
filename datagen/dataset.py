import pathlib

import pandas as pd
import re


def _mask(sentence: str, token: str, tok_pos: int, mask_token: str) -> str:
    assert token in sentence, f"{token} not found in {sentence}"
    # many = sentence.count(token)

    pattern = fr"\b{token}\b"

    approx_offset = tok_pos
    start = 0

    # Store offset of words in sentence
    offsets = [g.start(0) for g in re.finditer(pattern, sentence)]

    # Convert sentence offset to word index
    word_indices = [sentence[:offset].count(" ") for offset in offsets]

    closest, _ = min(zip(offsets, word_indices), key=lambda ow: abs(ow[1] - approx_offset))
    masked = f"{sentence[:closest]}{mask_token}{sentence[closest + len(token):]}"

    return masked


class SemCorDataSet:
    token_level: pd.DataFrame

    _EXPECTED_COLUMNS = (
        "id",
        "docid",
        "sntid",
        "tokid",
        "tokpos",
        "token",
        "lemma",
        "sense-keys",
    )

    def __init__(self, df: pd.DataFrame, mask_token=None):
        self.token_level = df
        self._check()
        self.token_level["sentence_idx"] = pd.factorize(
            self.token_level["docid"] + self.token_level["sntid"]
        )[0]
        if mask_token is not None:
            self.sentence_level = (
                self.token_level.groupby(["sentence_idx"])
                .agg({"token": " ".join})
                .rename(columns={"token": "sentence"})
                .reset_index()
            )
            maskable = pd.merge(
                self.token_level, self.sentence_level, on="sentence_idx", how="inner"
            )
            maskable = maskable[maskable["sense-keys"].notna()]
            maskable["masked"] = maskable[["sentence", "token", "tokpos"]].apply(
                lambda cols: _mask(*cols, mask_token), axis=1
            )
            self.masked = maskable[["masked", "token", "sense-keys", "sentence_idx"]]
        self.mask_token = mask_token

    @staticmethod
    def unpickle(inpath: pathlib.Path, mask_token=None) -> "SemCorDataSet":
        df = pd.read_pickle(inpath)
        return SemCorDataSet(df, mask_token)

    def pickle(self, out: pathlib.Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.token_level.to_pickle(out)

    def sense_keys(self, fullid: str) -> list:
        # dfid = f"{docid}.{sntid}.{tokid}"
        return self.token_level[self.token_level.id == fullid]["sense-keys"].split(",")

    def sentences(self) -> pd.DataFrame:
        return self.token_level.groupby(by=["docid", "sntid"])

    def _check(self):
        missing_columns = list(
            filter(
                lambda name: name not in self.token_level.columns,
                SemCorDataSet._EXPECTED_COLUMNS,
            )
        )
        if missing_columns:
            raise RuntimeError(
                f"Unable to construct {SemCorDataSet.__name__} from dataframe; missing column names {', '.join(missing_columns)}"
            )
