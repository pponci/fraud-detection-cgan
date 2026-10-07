# pyright: reportUnknownMemberType=false

from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray


def fit_classifier(
    x: pd.DataFrame, y: pd.Series, seed: int, params: dict[str, Any] | None = None
) -> LGBMClassifier:
    """
    LightGBM with library defaults unless params overrides them.
    """

    return LGBMClassifier(random_state=seed, **(params or {})).fit(x, y)


def predict_scores(
    clf: LGBMClassifier, x: pd.DataFrame
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Hard labels at LightGBM's default 0.5 threshold, and the fraud probability.
    """

    labels = cast(NDArray[np.int64], clf.predict(x))
    fraud_proba = cast(NDArray[np.float64], clf.predict_proba(x))[:, 1]

    return labels, fraud_proba
