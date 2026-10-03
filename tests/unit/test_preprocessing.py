from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import DatasetConfig
from fraud_detection.data.preprocessing import TabularPreprocessor
from fraud_detection.data.splitting import temporal_split


@pytest.fixture
def processed(
    raw: pd.DataFrame, cfg: DatasetConfig
) -> tuple[
    TabularPreprocessor, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]
]:
    train, val, test = temporal_split(raw)
    prep = TabularPreprocessor(cfg)

    return prep, prep.fit_transform(train), prep.transform(val), prep.transform(test), (train, val)


def test_no_missing_and_consistent_columns(
    processed: tuple[
        TabularPreprocessor,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> None:

    _, tr, va, te, _ = processed

    for df in (tr, va, te):
        assert not df.isna().any().any()

    assert tr.columns.tolist() == va.columns.tolist() == te.columns.tolist()
    assert "TransactionID" not in tr.columns and "TransactionDT" not in tr.columns
    assert {"day", "hour", "minute", "isFraud"} <= set(tr.columns)


def test_high_missing_columns_dropped(
    processed: tuple[
        TabularPreprocessor,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> None:
    _, tr, *_ = processed

    assert "R_emaildomain" not in tr.columns and "id_01" not in tr.columns


def test_row_filter_applies_to_train_only(
    processed: tuple[
        TabularPreprocessor,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> None:
    _, tr, va, _, (raw_train, raw_val) = processed

    assert len(tr) < len(raw_train)
    assert len(va) >= len(raw_val) - 1


def test_scaler_statistics_come_from_train(
    processed: tuple[
        TabularPreprocessor,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> None:
    prep, tr, va, *_ = processed
    scaled = prep.scaled_cols_

    assert np.allclose(tr[scaled].mean(), 0, atol=1e-8)
    assert not np.allclose(va[scaled].mean(), 0, atol=1e-3)


def test_transform_does_not_change_fitted_state(
    processed: tuple[
        TabularPreprocessor,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
    raw: pd.DataFrame,
) -> None:
    prep, *_ = processed
    chunk = raw.iloc[300:]
    before = prep.transform(chunk)
    prep.transform(raw.iloc[:100])

    pd.testing.assert_frame_equal(before, prep.transform(chunk))


def test_unseen_category_in_val_does_not_break_layout(
    raw: pd.DataFrame, cfg: DatasetConfig
) -> None:
    train, val, _ = temporal_split(raw)

    val.loc[val.index[0], "ProductCD"] = "Z"
    val.loc[val.index[1], "P_emaildomain"] = "brand-new.example"

    prep = TabularPreprocessor(cfg)
    tr = prep.fit_transform(train)
    va = prep.transform(val)

    assert va.columns.tolist() == tr.columns.tolist()


def test_save_load_roundtrip(
    processed: tuple[
        TabularPreprocessor,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
    raw: pd.DataFrame,
    tmp_path: Path,
) -> None:
    prep, *_ = processed
    prep.save(tmp_path / "prep.joblib")
    loaded = TabularPreprocessor.load(tmp_path / "prep.joblib")

    pd.testing.assert_frame_equal(prep.transform(raw.iloc[300:]), loaded.transform(raw.iloc[300:]))


def test_transform_before_fit_raises(raw: pd.DataFrame, cfg: DatasetConfig) -> None:
    with pytest.raises(RuntimeError):
        TabularPreprocessor(cfg).transform(raw)
