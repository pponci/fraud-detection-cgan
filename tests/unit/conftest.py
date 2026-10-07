import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import DatasetConfig, PreprocessingConfig, SplitConfig

CARD6 = ["debit", "credit", "charge card", "debit or credit"]
QUAL = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def with_nans(values: np.ndarray, p: float) -> pd.Series:
        s = pd.Series(values, dtype=object if isinstance(values[0], str) else float)

        return s.mask(rng.random(len(s)) < p)

    domains = np.array([f"mail{i}.com" for i in range(14)])

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(2987000, 2987000 + n),
            "isFraud": (rng.random(n) < 0.08).astype(int),
            "TransactionDT": np.sort(rng.integers(86400, 86400 * 60, n)),
            "TransactionAmt": rng.lognormal(4, 1.2, n).round(2),
            "ProductCD": rng.choice(list("WHCRS"), n),
            "card1": rng.integers(1000, 18000, n).astype(float),
            "card2": with_nans(rng.integers(100, 600, n).astype(float), 0.02),
            "card3": with_nans(rng.choice([150.0, 185.0, 100.0], n), 0.01),
            "card4": with_nans(rng.choice(["visa", "mastercard", "discover"], n), 0.01),
            "card5": with_nans(rng.choice([226.0, 224.0, 166.0], n), 0.01),
            "card6": with_nans(rng.choice(CARD6, n), 0.01),
            "addr1": with_nans(rng.integers(100, 540, n).astype(float), 0.1),
            "addr2": with_nans(rng.choice([87.0, 60.0, 96.0], n), 0.1),
            "P_emaildomain": with_nans(rng.choice(domains, n), 0.15),
            "R_emaildomain": with_nans(rng.choice(domains, n), 0.75),
            "M1": with_nans(rng.choice(["T", "F"], n), 0.3),
            "M2": with_nans(rng.choice(["T", "F"], n), 0.3),
            "M4": with_nans(rng.choice(["M0", "M1", "M2"], n), 0.4),
            "id_01": with_nans(rng.normal(-5, 3, n), 0.9),
        }
    )
    for i in range(1, 4):
        df[f"C{i}"] = rng.poisson(2, n).astype(float)
        df[f"D{i}"] = with_nans(rng.exponential(30, n), 0.3)

    for i in range(1, 7):
        levels = rng.choice([0.0, 1.0, 2.0, 5.0, 40.0], n, p=[0.6, 0.2, 0.1, 0.08, 0.02])
        df[f"V{i}"] = with_nans(levels, 0.2)

    df["Vbin"] = with_nans(rng.integers(0, 2, n).astype(float), 0.2)
    sparse = df.columns.difference(["TransactionID", "TransactionDT"])
    df.loc[5, sparse] = np.nan

    return df


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def cfg() -> DatasetConfig:
    return DatasetConfig(
        name="synthetic",
        loader="ieee_cis",
        raw_files={},
        target="isFraud",
        time_col="TransactionDT",
        id_cols=["TransactionID"],
        split=SplitConfig(),
        preprocessing=PreprocessingConfig(
            qualitative_cols=QUAL,
            already_encoded_cols=["addr1", "addr2", "card1", "card2", "card3", "card5"],
        ),
    )


@pytest.fixture
def toy_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Small learnable fraud-like problem with an integer dummy column (~10% fraud).
    """

    rng = np.random.default_rng(0)

    def make(n: int) -> pd.DataFrame:
        df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
        df["dummy"] = rng.integers(0, 2, n)
        score = df["f0"] + 0.5 * df["f1"] + rng.normal(scale=0.7, size=n)
        df["isFraud"] = (score > np.quantile(score, 0.9)).astype(int)

        return df

    return make(1500), make(600)
