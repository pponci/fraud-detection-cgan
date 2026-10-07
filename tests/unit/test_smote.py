import pandas as pd
import pytest

from fraud_detection.augmentation.smote import add_synthetic, smote_synthetic


def test_adds_exactly_enough_rows_for_the_ratio(
    toy_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = toy_split
    x, y = train.drop(columns="isFraud"), train["isFraud"]
    n_fraud, n_legit = int(y.sum()), int((y == 0).sum())

    synthetic = smote_synthetic(x, y, ratio=0.5, seed=1)

    assert len(synthetic) == round(0.5 * n_legit) - n_fraud
    assert synthetic.columns.tolist() == x.columns.tolist()


def test_integer_columns_stay_integer(toy_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = toy_split
    x, y = train.drop(columns="isFraud"), train["isFraud"]
    synthetic = smote_synthetic(x, y, ratio=0.3, seed=1)

    assert synthetic["dummy"].dtype == x["dummy"].dtype
    assert set(synthetic["dummy"].unique()) <= {0, 1}


def test_same_seed_same_samples(toy_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = toy_split
    x, y = train.drop(columns="isFraud"), train["isFraud"]

    pd.testing.assert_frame_equal(
        smote_synthetic(x, y, 0.3, seed=5), smote_synthetic(x, y, 0.3, seed=5)
    )


def test_add_synthetic_appends_labelled_rows(toy_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = toy_split
    x, y = train.drop(columns="isFraud"), train["isFraud"]
    synthetic = smote_synthetic(x, y, ratio=0.3, seed=1)

    x_aug, y_aug = add_synthetic(x, y, synthetic)

    assert len(x_aug) == len(y_aug) == len(x) + len(synthetic)
    assert x_aug.iloc[: len(x)].reset_index(drop=True).equals(x.reset_index(drop=True))
    assert (y_aug.iloc[len(x) :] == 1).all()
    assert y_aug.dtype == y.dtype


def test_ratio_below_current_prevalence_is_a_clear_error(
    toy_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = toy_split
    x, y = train.drop(columns="isFraud"), train["isFraud"]

    with pytest.raises(ValueError, match="can only add"):
        smote_synthetic(x, y, ratio=0.05, seed=1)
