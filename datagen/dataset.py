import pathlib

import pandas as pd


class SemCorDataSet:
    token_level: pd.DataFrame

    _EXPECTED_COLUMNS = (
        "id",
        "docid",
        "sntid",
        "tokid",
        "token",
        "lemma",
        "sense-keys",
    )

    def __init__(self, df: pd.DataFrame):
        self.token_level = df
        self._check()
        self.token_level["sentence_idx"] = pd.factorize(self.token_level["docid"] + self.token_level["sntid"])[0]
        self.sentence_level =  (
            self.token_level.groupby(["sentence_idx"])
            .agg({"token": " ".join})
            .rename(columns={"token": "sentence"})
        )
        

    @staticmethod
    def unpickle(inpath: pathlib.Path) -> "SemCorDataSet":
        df = pd.read_pickle(inpath)
        return SemCorDataSet(df)

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
            filter(lambda name: name not in self.token_level.columns, SemCorDataSet._EXPECTED_COLUMNS)
        )
        if missing_columns:
            raise RuntimeError(
                f"Unable to construct {SemCorDataSet.__name__} from dataframe; missing column names {', '.join(missing_columns)}"
            )
