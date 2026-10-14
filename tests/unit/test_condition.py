import numpy as np
import pandas as pd
import pytest

from fraud_detection.augmentation.condition import sample_condition_scores, training_conditions


def make_table() -> tuple[pd.DataFrame, pd.Series]:
    """
    10 rows whose score equals the row number; rows 2, 5 and 8 are fraud.
    """

    scores = pd.DataFrame({"ae_score": np.arange(10, dtype=np.float32)})
    y = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

    return scores, y


def test_own_score_pairs_each_fraud() -> None:
    scores, y = make_table()
    condition = training_conditions(scores, y)

    assert condition.shape == (3, 1) and condition.dtype == np.float32
    assert condition[:, 0].tolist() == [2.0, 5.0, 8.0]


def test_mismatched_lengths_and_unknown_pairing_are_rejected() -> None:
    scores, y = make_table()

    with pytest.raises(ValueError, match="scores for"):
        training_conditions(scores.iloc[:5], y)


@pytest.fixture
def population() -> tuple[pd.DataFrame, pd.Series]:
    """
    Legitimate rows score low; 100 frauds split 90 low / 10 high; one extreme outlier.
    """

    rng = np.random.default_rng(0)
    legit = rng.normal(0.0, 0.5, 1000)
    fraud_low = rng.normal(0.5, 0.2, 90)
    fraud_high = rng.normal(5.0, 0.2, 10)
    values = np.concatenate([legit, fraud_low, fraud_high, [10.0]])
    y = np.concatenate([np.zeros(1000), np.ones(100), [0]]).astype(int)

    return pd.DataFrame({"ae_score": values.astype(np.float32)}), pd.Series(y)


def test_samples_come_from_the_real_fraud_scores(
    population: tuple[pd.DataFrame, pd.Series],
) -> None:
    scores, y = population
    sampled = sample_condition_scores(scores, y, n_rows=500, seed=1)

    assert sampled.shape == (500, 1) and sampled.dtype == np.float32

    fraud_scores = set(scores["ae_score"].to_numpy()[y.to_numpy() == 1])

    assert set(sampled[:, 0]) <= fraud_scores


def test_the_requested_number_of_rows_is_exact_despite_rounding(
    population: tuple[pd.DataFrame, pd.Series],
) -> None:
    scores, y = population

    for n_rows in (1, 7, 333, 1001):
        assert len(sample_condition_scores(scores, y, n_rows, seed=2)) == n_rows


def test_bin_shares_of_the_real_frauds_are_preserved(
    population: tuple[pd.DataFrame, pd.Series],
) -> None:
    scores, y = population
    sampled = sample_condition_scores(scores, y, n_rows=2000, seed=3)[:, 0]

    assert (sampled > 2.5).mean() == pytest.approx(0.10, abs=0.01)


def test_same_seed_same_scores_different_seed_different(
    population: tuple[pd.DataFrame, pd.Series],
) -> None:
    scores, y = population
    first = sample_condition_scores(scores, y, 200, seed=4)

    assert np.array_equal(first, sample_condition_scores(scores, y, 200, seed=4))
    assert not np.array_equal(first, sample_condition_scores(scores, y, 200, seed=5))
