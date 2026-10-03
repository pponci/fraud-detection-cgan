from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.config import load_dataset_config
from fraud_detection.data.datasets import load_raw


def _write_raw(root: Path) -> None:
    cfg = load_dataset_config("ieee_cis", data_root=root)
    cfg.raw_dir.mkdir(parents=True)

    pd.DataFrame(
        {"TransactionID": [3, 1, 2], "TransactionDT": [300, 100, 200], "isFraud": [0, 1, 0]}
    ).to_csv(cfg.raw_dir / "train_transaction.csv", index=False)

    pd.DataFrame({"TransactionID": [1, 2], "id_01": [-5.0, -10.0]}).to_csv(
        cfg.raw_dir / "train_identity.csv", index=False
    )


def test_left_join_keeps_transactions_without_identity(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    raw = load_raw(load_dataset_config("ieee_cis", data_root=tmp_path))

    assert len(raw) == 3
    assert raw.loc[raw["TransactionID"] == 3, "id_01"].isna().all()


def test_output_is_sorted_by_time_with_clean_index(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    raw = load_raw(load_dataset_config("ieee_cis", data_root=tmp_path))

    assert raw["TransactionDT"].tolist() == [100, 200, 300]
    assert raw.index.tolist() == [0, 1, 2]


def test_unknown_loader_is_a_clear_error(tmp_path: Path) -> None:
    cfg = replace(load_dataset_config("ieee_cis", data_root=tmp_path), loader="nope")

    with pytest.raises(KeyError, match="Unknown loader"):
        load_raw(cfg)
