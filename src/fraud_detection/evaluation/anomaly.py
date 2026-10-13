# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from typing import Self, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from fraud_detection.training.rows import RowDataset


def reconstruction_errors(
    model: nn.Module, x: pd.DataFrame, device: torch.device, batch_size: int = 256
) -> NDArray[np.float32]:
    """
    Per-row mean squared reconstruction error, in the row order of x.
    """

    loader = DataLoader(RowDataset(x), batch_size=batch_size, shuffle=False)
    model.eval()
    errors: list[NDArray[np.float32]] = []

    with torch.no_grad():
        for real in loader:
            real = real.to(device)
            batch_error = torch.mean((model(real) - real) ** 2, dim=1)
            errors.append(batch_error.cpu().numpy())

    return np.concatenate(errors)


def error_summary(
    errors: NDArray[np.float32], y: pd.Series, dataset: str
) -> dict[str, float | str]:
    """
    Mean and median error per class, and the separation rate (fraud mean / legit mean).
    """

    is_legit = y.to_numpy() == 0
    legit, fraud = errors[is_legit], errors[~is_legit]

    return {
        "dataset": dataset,
        "n_mean": float(legit.mean()),
        "f_mean": float(fraud.mean()),
        "n_median": float(np.median(legit)),
        "f_median": float(np.median(fraud)),
        "sep_rate": float(fraud.mean() / legit.mean()),
    }


class AnomalyScorer:
    """
    Error -> anomaly score: log(1 + error), then standardised on the TRAINING errors.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    @staticmethod
    def _log(errors: NDArray[np.float32]) -> NDArray[np.float32]:
        return cast(NDArray[np.float32], np.log1p(np.maximum(errors, 0)).reshape(-1, 1))

    def fit(self, errors: NDArray[np.float32]) -> Self:
        self.scaler.fit(self._log(errors))
        return self

    def transform(self, errors: NDArray[np.float32]) -> pd.DataFrame:
        scores = cast(NDArray[np.float32], self.scaler.transform(self._log(errors)))
        return pd.DataFrame(scores, columns=["ae_score"])
