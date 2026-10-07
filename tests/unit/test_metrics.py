import pytest

from fraud_detection.evaluation.metrics import METRIC_NAMES, classification_metrics


def test_known_values() -> None:
    scores = classification_metrics(
        y_true=[0, 0, 1, 1], y_pred=[0, 1, 1, 1], y_prob=[0.1, 0.6, 0.8, 0.9]
    )

    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(0.8)
    assert scores["roc_auc"] == pytest.approx(1.0)
    assert scores["pr_auc"] == pytest.approx(1.0)


def test_returns_exactly_the_reported_metrics() -> None:
    scores = classification_metrics([0, 1, 0, 1], [0, 1, 1, 0], [0.2, 0.7, 0.6, 0.4])

    assert tuple(scores) == METRIC_NAMES
    assert all(isinstance(value, float) for value in scores.values())
