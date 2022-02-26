import pathlib

import pandas as pd


class SemCorDataSet:
    df: pd.DataFrame

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
        missing_columns = list(
            filter(lambda name: name not in df.columns, SemCorDataSet._EXPECTED_COLUMNS)
        )
        if missing_columns:
            raise RuntimeError(
                f"Unable to construct {SemCorDataSet.__name__} from dataframe; missing column names {', '.join(missing_columns)}"
            )

        self.df = df

    @staticmethod
    def unpickle(inpath: pathlib.Path) -> "SemCorDataSet":
        df = pd.read_pickle(inpath)
        return SemCorDataSet(df)

    def pickle(self, out: pathlib.Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(out)

    def sentences(self) -> pd.DataFrame:
        return self.df.groupby(by=["sntid"])
