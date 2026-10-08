from collections.abc import Callable
from typing import Any

import pandas as pd

from fraud_detection.augmentation.smote import add_synthetic
from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.evaluation.synthetic import synthetic_quality
from fraud_detection.training.classifier import fit_classifier, predict_scores


def score_model(
    method: str,
    ratio: float,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Fit on (x_fit, y_fit) and score on it and on validation.
    """

    clf = fit_classifier(x_fit, y_fit, seed, params)

    rows: list[dict[str, Any]] = []

    for dataset, x, y in (("train", x_fit, y_fit), ("val", x_val, y_val)):
        labels, fraud_proba = predict_scores(clf, x)
        scores = classification_metrics(y, labels, fraud_proba)
        rows.append({"method": method, "ratio": ratio, "dataset": dataset, **scores})

    return rows


def evaluate_synthetic_sweep(
    method: str,
    ratios: list[float],
    make_synthetic: Callable[[float], pd.DataFrame],
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    seed: int,
    classifier_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (classification metrics, synthetic-sample quality), one entry per ratio.
    """

    x_train, y_train = train.drop(columns=target), train[target]
    x_val, y_val = val.drop(columns=target), val[target]
    real_fraud = x_train[y_train == 1]

    metric_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for ratio in ratios:
        synthetic = make_synthetic(ratio)

        quality_rows.append(
            {"method": method, "ratio": ratio, **synthetic_quality(real_fraud, synthetic)}
        )

        x_aug, y_aug = add_synthetic(x_train, y_train, synthetic)

        metric_rows += score_model(
            method, ratio, x_aug, y_aug, x_val, y_val, seed, classifier_params
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(quality_rows)
