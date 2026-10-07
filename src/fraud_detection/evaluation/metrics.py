# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from typing import SupportsFloat

import sklearn.metrics as skm
from numpy.typing import ArrayLike

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def _scalar(value: SupportsFloat) -> float:
    return float(value)


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_prob: ArrayLike
) -> dict[str, float]:
    """
    Threshold metrics use y_pred; ranking metrics (ROC-AUC, PR-AUC) use y_prob.
    """
    return {
        "accuracy": _scalar(skm.accuracy_score(y_true, y_pred)),
        "precision": _scalar(skm.precision_score(y_true, y_pred)),
        "recall": _scalar(skm.recall_score(y_true, y_pred)),
        "f1": _scalar(skm.f1_score(y_true, y_pred)),
        "roc_auc": _scalar(skm.roc_auc_score(y_true, y_prob)),
        "pr_auc": _scalar(skm.average_precision_score(y_true, y_prob)),
    }
