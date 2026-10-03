from collections.abc import Callable

import pandas as pd

from fraud_detection.config import DatasetConfig


def _load_ieee_cis(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Left-join identity onto transactions: not every transaction has an identity row.
    """

    transaction = pd.read_csv(cfg.raw_dir / cfg.raw_files["transaction"])
    identity = pd.read_csv(cfg.raw_dir / cfg.raw_files["identity"])

    return transaction.merge(identity, how="left", on="TransactionID")


LOADERS: dict[str, Callable[[DatasetConfig], pd.DataFrame]] = {
    "ieee_cis": _load_ieee_cis,
}


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Load the raw table for a dataset, sorted by time.
    """

    if cfg.loader not in LOADERS:
        raise KeyError(f"Unknown loader {cfg.loader!r}. Registered: {sorted(LOADERS)}")

    df = LOADERS[cfg.loader](cfg)

    return df.sort_values(by=cfg.time_col).reset_index(drop=True)
