import numpy as np
import pandas as pd
from numpy.typing import NDArray


def training_conditions(scores: pd.DataFrame, y_train: pd.Series) -> NDArray[np.float32]:
    """
    The condition shown with each real fraud sample during training, shape (n_fraud, 1).
    """

    if len(scores) != len(y_train):
        raise ValueError(f"{len(scores)} scores for {len(y_train)} training rows")

    is_fraud = y_train.to_numpy() == 1
    column = scores["ae_score"].to_numpy(dtype=np.float32)

    chosen = column[is_fraud]

    return chosen.reshape(-1, 1)


def sample_condition_scores(
    scores: pd.DataFrame, y_train: pd.Series, n_rows: int, seed: int, n_bins: int = 3
) -> NDArray[np.float32]:
    """
    Anomaly scores to condition ``n_rows`` generated samples on, shape (n_rows, 1).
    """

    all_scores = scores["ae_score"]
    bins = np.asarray(pd.cut(all_scores, bins=n_bins, labels=False))
    is_fraud = y_train.to_numpy() == 1
    fraud_scores = all_scores.to_numpy(dtype=np.float32)[is_fraud]
    fraud_bins = bins[is_fraud]

    bin_values, counts = np.unique(fraud_bins, return_counts=True)
    per_bin = np.round(counts / counts.sum() * n_rows).astype(int)
    remainder = n_rows - int(per_bin.sum())

    if remainder != 0:
        per_bin[int(np.argmax(per_bin))] += remainder

    rng = np.random.RandomState(seed)
    picked: list[int] = []

    for bin_value, count in zip(bin_values, per_bin, strict=True):
        members = np.flatnonzero(fraud_bins == bin_value)
        picked.extend(rng.choice(members, size=count, replace=True))

    rng.shuffle(picked)

    return fraud_scores[picked].reshape(-1, 1)
