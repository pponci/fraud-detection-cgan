import pandas as pd

from fraud_detection.evaluation.metrics import METRIC_NAMES
from fraud_detection.training.benchmark import run_benchmark

PARAMS = {"n_estimators": 20, "n_jobs": 1, "verbose": -1}


def test_layout_and_ranges(toy_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, val = toy_split
    metrics, synthetic = run_benchmark(train, val, "isFraud", [0.2, 0.5], 3, PARAMS)

    assert len(metrics) == 6
    assert set(metrics["dataset"]) == {"train", "val"}
    assert metrics[metrics["method"] == "base"]["ratio"].isna().all()
    assert metrics[metrics["method"] == "smote"]["ratio"].nunique() == 2
    assert ((metrics[list(METRIC_NAMES)] >= 0) & (metrics[list(METRIC_NAMES)] <= 1)).all().all()
    assert len(synthetic) == 2
    assert {"Wasserstein", "Corr_diff", "NN_real_to_fake", "NN_fake_to_real"} <= set(
        synthetic.columns
    )


def test_learns_something(toy_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, val = toy_split
    metrics, _ = run_benchmark(train, val, "isFraud", [0.3], 3, PARAMS)
    base_val = metrics[(metrics["method"] == "base") & (metrics["dataset"] == "val")]

    assert base_val["roc_auc"].iloc[0] > 0.8


def test_deterministic_for_a_seed(toy_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, val = toy_split
    first = run_benchmark(train, val, "isFraud", [0.3], 7, PARAMS)
    second = run_benchmark(train, val, "isFraud", [0.3], 7, PARAMS)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
