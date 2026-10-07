import numpy as np
import pandas as pd
import pytest

from fraud_detection.evaluation.synthetic import (
    corr_distance,
    nn_distance,
    synthetic_quality,
    wasserstein_dist,
)


@pytest.fixture
def real() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)

    return pd.DataFrame(
        {"a": base, "b": base + rng.normal(scale=0.3, size=300), "c": rng.normal(size=300)}
    )


def test_identical_data_scores_zero(real: pd.DataFrame) -> None:
    scores = synthetic_quality(real, real.copy())

    assert scores == {
        "Wasserstein": 0.0,
        "Corr_diff": 0.0,
        "NN_real_to_fake": 0.0,
        "NN_fake_to_real": 0.0,
    }


def test_shifted_marginal_is_measured(real: pd.DataFrame) -> None:
    shifted = real.copy()
    shifted["a"] += 2.0  # one of three columns moves by exactly 2
    assert wasserstein_dist(real, shifted) == pytest.approx(2.0 / 3)


def test_broken_correlation_is_measured(real: pd.DataFrame) -> None:
    shuffled = real.copy()
    shuffled["b"] = np.random.default_rng(2).permutation(shuffled["b"].to_numpy())
    assert corr_distance(real, shuffled) > 0.05


def test_nn_distance_grows_with_displacement(real: pd.DataFrame) -> None:
    dist_to_real_large, dist_to_fake_large = nn_distance(real, real + 5.0)
    dist_to_real_small, dist_to_fake_small = nn_distance(real, real + 0.1)

    assert dist_to_real_large > dist_to_real_small > 0
    assert dist_to_fake_large > dist_to_fake_small > 0


def test_constant_columns_do_not_break_corr_distance(real: pd.DataFrame) -> None:
    real = real.assign(const=0.0)
    assert corr_distance(real, real.copy()) == 0.0
