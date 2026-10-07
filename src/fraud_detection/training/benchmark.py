from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.augmentation.smote import add_synthetic, smote_synthetic
from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.evaluation.synthetic import synthetic_quality
from fraud_detection.training.classifier import fit_classifier, predict_scores


def _score(
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
    The train rows are scored on the data the model was fitted on, which
    for SMOTE includes the synthetic samples. Only the val rows are comparable across methods.
    """

    clf = fit_classifier(x_fit, y_fit, seed, params)
    rows: list[dict[str, Any]] = []

    for dataset, x, y in (("train", x_fit, y_fit), ("val", x_val, y_val)):
        labels, fraud_proba = predict_scores(clf, x)
        scores = classification_metrics(y, labels, fraud_proba)
        rows.append({"method": method, "ratio": ratio, "dataset": dataset, **scores})

    return rows


def run_benchmark(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    ratios: list[float],
    seed: int,
    classifier_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (classification metrics, synthetic-sample quality), one row per setting.
    """

    x_train, y_train = train.drop(columns=target), train[target]
    x_val, y_val = val.drop(columns=target), val[target]

    metric_rows = _score("base", np.nan, x_train, y_train, x_val, y_val, seed, classifier_params)
    synth_rows: list[dict[str, Any]] = []

    real_fraud = x_train[y_train == 1]

    for ratio in ratios:
        synthetic = smote_synthetic(x_train, y_train, ratio, seed)
        synth_rows.append(
            {"method": "smote", "ratio": ratio, **synthetic_quality(real_fraud, synthetic)}
        )
        x_aug, y_aug = add_synthetic(x_train, y_train, synthetic)
        metric_rows += _score("smote", ratio, x_aug, y_aug, x_val, y_val, seed, classifier_params)

    return pd.DataFrame(metric_rows), pd.DataFrame(synth_rows)
