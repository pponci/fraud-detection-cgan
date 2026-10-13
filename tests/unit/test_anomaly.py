import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fraud_detection.evaluation.anomaly import AnomalyScorer, error_summary, reconstruction_errors

CPU = torch.device("cpu")


class Shift(nn.Module):
    """
    A model that reconstructs every row as row + 1, so every error is exactly 1.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def test_reconstruction_error_is_the_per_row_mean_squared_difference() -> None:
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 4)))
    errors = reconstruction_errors(Shift(), x, CPU)

    assert errors.shape == (10,)
    assert np.allclose(errors, 1.0)


def test_errors_keep_row_order_across_batches() -> None:
    class Square(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    x = pd.DataFrame(np.random.default_rng(1).normal(size=(300, 3)))
    errors = reconstruction_errors(Square(), x, CPU, batch_size=64)
    expected = (x.to_numpy(dtype=np.float32) ** 2).mean(axis=1)

    assert errors.shape == (300,)
    assert np.allclose(errors, expected, atol=1e-6)


def test_error_summary_separates_the_classes() -> None:
    errors = np.array([1.0, 1.0, 3.0, 5.0], dtype=np.float32)
    y = pd.Series([0, 0, 1, 1])
    summary = error_summary(errors, y, "train")

    assert summary == {
        "dataset": "train",
        "n_mean": 1.0,
        "f_mean": 4.0,
        "n_median": 1.0,
        "f_median": 4.0,
        "sep_rate": 4.0,
    }


def test_scores_are_standardised_on_the_training_errors() -> None:
    train = np.random.default_rng(0).exponential(size=2000).astype(np.float32)
    scorer = AnomalyScorer().fit(train)
    scores = scorer.transform(train)

    assert list(scores.columns) == ["ae_score"]
    assert scores["ae_score"].dtype == np.float32
    assert scores["ae_score"].mean() == pytest.approx(0.0, abs=1e-4)
    assert scores["ae_score"].std(ddof=0) == pytest.approx(1.0, abs=1e-4)


def test_scores_increase_with_the_error() -> None:
    scorer = AnomalyScorer().fit(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    scores = scorer.transform(np.array([0.05, 0.5, 5.0], dtype=np.float32))["ae_score"]

    assert scores.is_monotonic_increasing


def test_the_scale_comes_from_train_only() -> None:
    scorer = AnomalyScorer().fit(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    shifted = scorer.transform(np.array([5.0, 6.0, 7.0], dtype=np.float32))["ae_score"]

    assert shifted.mean() > 5


def test_a_tiny_negative_error_is_clipped_not_nan() -> None:
    scorer = AnomalyScorer().fit(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    scores = scorer.transform(np.array([-1e-9], dtype=np.float32))

    assert not scores.isna().any().any()
