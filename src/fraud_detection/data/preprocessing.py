# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# joblib and scikit-learn ship no type stubs. This relaxes only these two rules, only in this file.

from pathlib import Path
from typing import Self, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import DatasetConfig
from fraud_detection.data.cleaning import clean_column_names, missing_summary

BOOL_MAP = {"T": 1, "F": 0}
SECONDS_IN_DAY = 24 * 60 * 60
SECONDS_IN_HOUR = 60 * 60


class TabularPreprocessor:
    """
    Learns cleaning/encoding/scaling on train; transform replays it on other splits.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.is_fitted = False

    def _fill_and_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, median in self.medians_.items():
            df[col] = df[col].fillna(median)

        for col, mode in self.modes_.items():
            df[col] = df[col].fillna(mode)

        for col in self.encoded_int_cols_:
            df[col] = df[col].fillna(1)  # existing codes start well above 1, so 1 = "missing"

        for col in self.cat_cols_:
            df[col] = df[col].fillna("unknown")

        df = df.dropna(subset=[self.cfg.target, self.cfg.time_col])

        return df.drop_duplicates()

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        c = self.cfg.preprocessing
        df = df.copy()

        for col in self.bool_cols_:
            unexpected = set(df[col].unique()) - set(BOOL_MAP)

            if unexpected:
                raise ValueError(
                    f"{col!r} has <= 2 unique values and is treated as boolean, but contains "
                    f"{sorted(map(str, unexpected))} instead of 'T'/'F'."
                )

            df[col] = df[col].map(BOOL_MAP).astype(int)

        for col in self.encoded_int_cols_:
            df[col] = df[col].astype(int)

        if fit:
            self.ohe_cols_ = [col for col in self.cat_cols_ if df[col].nunique() < c.ohe_max_unique]
            self.label_cols_ = [col for col in self.cat_cols_ if col not in self.ohe_cols_]

            df = pd.get_dummies(df, columns=self.ohe_cols_, dtype=int)
            self.ohe_columns_ = df.columns.tolist()

            df = clean_column_names(df)
            self.label_maps_ = {
                col: {cat: code for code, cat in enumerate(sorted(df[col].unique()), start=1)}
                for col in self.label_cols_
            }

        else:
            df = pd.get_dummies(df, columns=self.ohe_cols_, dtype=int)
            df = df.reindex(columns=self.ohe_columns_, fill_value=0)
            df = clean_column_names(df)

        for col, mapping in self.label_maps_.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").map(mapping).fillna(0)

        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = self.cfg.preprocessing.time_features
        col = self.cfg.time_col

        if mode not in ("seconds_offset", "drop"):
            raise ValueError(f"Unknown time_features mode {mode!r}")

        df = df.copy()

        if mode == "seconds_offset":
            df["day"] = (df[col] // SECONDS_IN_DAY).astype(int)
            df["hour"] = ((df[col] % SECONDS_IN_DAY) // SECONDS_IN_HOUR).astype(int)
            df["minute"] = ((df[col] % SECONDS_IN_HOUR) // 60).astype(int)

        return df.drop(columns=col)

    def _skew(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        if fit:
            skew = df[self.quant_cols_].skew()
            threshold = self.cfg.preprocessing.skew_threshold

            self.skewed_cols_: list[str] = [str(col) for col in skew.index[skew.abs() > threshold]]

            mins = df[self.skewed_cols_].min()

            self.shifts_: dict[str, float] = {
                col: 0.0 if mins[col] >= 0 else abs(float(mins[col])) + 1
                for col in self.skewed_cols_
            }

        for col, shift in self.shifts_.items():
            df[col] = df[col] + shift

        for col in self.skewed_cols_:
            df[col] = np.log1p(np.maximum(df[col], 0))

        return df

    def _scale(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            is_binary = {col: set(df[col].dropna().unique()).issubset({0, 1}) for col in df.columns}

            self.unscaled_cols_ = [col for col in df.columns if is_binary[col]]
            self.scaled_cols_ = [col for col in df.columns if not is_binary[col]]

            self.scaler_ = StandardScaler().fit(df[self.scaled_cols_])

        values = cast(NDArray[np.float64], self.scaler_.transform(df[self.scaled_cols_]))
        scaled = pd.DataFrame(values, columns=self.scaled_cols_)

        return pd.concat([scaled, df[self.unscaled_cols_].reset_index(drop=True)], axis=1)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg.preprocessing

        col_summary = missing_summary(train, "cols")
        self.kept_columns_ = col_summary.loc[
            col_summary["p_missing"] <= c.max_col_missing, "name"
        ].tolist()
        df = train[self.kept_columns_]

        if not df.index.is_unique:
            raise ValueError("Train index must be unique (row filtering selects by label).")

        row_summary = missing_summary(df, "rows")
        df = df.loc[row_summary.loc[row_summary["p_missing"] <= c.max_row_missing, "name"]]

        df = df.drop(columns=[col for col in self.cfg.id_cols if col in df.columns])

        cols = [col for col in df.columns if col not in (self.cfg.time_col, self.cfg.target)]
        self.qual_cols_ = [col for col in cols if col in c.qualitative_cols]
        self.quant_cols_ = [col for col in cols if col not in self.qual_cols_]
        self.medians_ = {col: df[col].median() for col in self.quant_cols_}
        self.bool_cols_ = [col for col in self.qual_cols_ if df[col].nunique() <= 2]
        self.modes_ = {col: df[col].mode().iloc[0] for col in self.bool_cols_}
        self.encoded_int_cols_ = [col for col in c.already_encoded_cols if col in df.columns]
        self.cat_cols_ = [
            col
            for col in self.qual_cols_
            if col not in self.encoded_int_cols_ and col not in self.bool_cols_
        ]
        df = self._fill_and_dedup(df)

        df = self._encode(df, fit=True)
        df = self._time_features(df)
        df = self._skew(df, fit=True)
        out = self._scale(df, fit=True)
        self.is_fitted = True

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform on the training split first.")

        df = df[self.kept_columns_]
        df = df.drop(columns=[col for col in self.cfg.id_cols if col in df.columns])
        df = self._fill_and_dedup(df)
        df = self._encode(df, fit=False)
        df = self._time_features(df)
        df = self._skew(df, fit=False)

        return self._scale(df, fit=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        obj = joblib.load(path)

        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")

        return obj
