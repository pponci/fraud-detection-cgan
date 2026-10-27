import numpy as np
import pytest

from fraud_detection.evaluation.ranking import (
    RECALL_TARGETS,
    false_alerts,
    precision_at_recall,
    ranking_metrics,
)


def test_ranking_metrics_on_the_classic_four_point_example() -> None:
    scores = ranking_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])

    assert scores["pr_auc"] == pytest.approx(5 / 6)
    assert scores["roc_auc"] == pytest.approx(0.75)
    assert list(scores) == [
        "pr_auc",
        "roc_auc",
        "precision_at_80_recall",
        "precision_at_85_recall",
        "precision_at_90_recall",
        "precision_at_95_recall",
        "precision_at_100_recall",
    ]


def test_precision_at_recall_reads_the_closest_recall_and_rounds() -> None:
    recalls = np.array([1.0, 0.8, 0.5, 0.0])
    precisions = np.array([0.30001, 0.6666, 0.9, 1.0])

    assert precision_at_recall(recalls, precisions, 0.78) == 0.667
    assert precision_at_recall(recalls, precisions, 0.45) == 0.9


def test_precision_at_recall_ties_go_to_the_first_lowest_threshold_point() -> None:
    recalls = np.array([1.0, 1.0, 0.5, 0.0])
    precisions = np.array([0.4, 0.6, 0.9, 1.0])

    assert precision_at_recall(recalls, precisions, 1.0) == 0.4


def test_false_alerts_by_hand() -> None:
    y = [0, 0, 1, 1, 0, 1]
    proba = [0.1, 0.2, 0.9, 0.8, 0.6, 0.7]

    assert false_alerts([1.0, 0.5, 0.3, 0.05], y, proba) == [3, 0, 0, 0]


def test_false_alerts_never_decrease_when_more_fraud_must_be_caught() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    proba = rng.random(2000) * 0.5 + 0.4 * y
    alerts = false_alerts(RECALL_TARGETS, y, proba)

    assert len(alerts) == 20 and alerts == sorted(alerts)


def test_the_recall_targets_are_5_to_100_percent_in_20_steps() -> None:
    assert RECALL_TARGETS[0] == 0.05 and RECALL_TARGETS[-1] == 1.0 and len(RECALL_TARGETS) == 20
