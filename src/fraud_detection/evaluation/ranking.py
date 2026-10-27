# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from typing import Any

import numpy as np
import sklearn.metrics as skm
from numpy.typing import ArrayLike, NDArray


def _scalar(value: Any) -> float:  # noqa: ANN401
    return float(value)


RECALL_LEVELS = (0.8, 0.85, 0.9, 0.95, 1.0)
RECALL_TARGETS = [round(float(x), 2) for x in np.linspace(0.05, 1, 20)]


def precision_at_recall(
    recalls: NDArray[np.float64], precisions: NDArray[np.float64], target_recall: float
) -> float:
    """
    Precision at the point of the PR curve whose recall is closest to target_recall.
    """

    idx = int(np.argmin(np.abs(recalls - target_recall)))

    return round(float(precisions[idx]), 3)


def ranking_metrics(y_true: ArrayLike, y_proba: ArrayLike) -> dict[str, float]:
    """
    PR-AUC (average precision), ROC-AUC and precision at 80/85/90/95/100% recall.
    """

    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    precision, recall, _ = skm.precision_recall_curve(y_true, y_proba)
    scores = {
        "pr_auc": _scalar(skm.average_precision_score(y_true, y_proba)),
        "roc_auc": _scalar(skm.roc_auc_score(y_true, y_proba)),
    }

    for level in RECALL_LEVELS:
        name = f"precision_at_{round(level * 100)}_recall"
        scores[name] = precision_at_recall(recall, precision, level)

    return scores


def false_alerts(recall_targets: list[float], y_true: ArrayLike, y_proba: ArrayLike) -> list[int]:
    """
    Legitimate transactions flagged when the threshold is set to catch each target recall.
    """

    truth: NDArray[np.int64] = np.asarray(y_true)
    proba: NDArray[np.float64] = np.asarray(y_proba)
    recalls: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    _, recalls, thresholds = skm.precision_recall_curve(truth, proba)

    alerts: list[int] = []

    for target in recall_targets:
        idx = min(int(np.argmin(np.abs(recalls - target))), len(thresholds) - 1)
        flagged = proba >= thresholds[idx]
        alerts.append(int((flagged & (truth == 0)).sum()))

    return alerts
