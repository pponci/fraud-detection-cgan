# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from typing import cast

import pandas as pd
from imblearn.over_sampling import SMOTE


def smote_synthetic(
    x_train: pd.DataFrame, y_train: pd.Series, ratio: float, seed: int
) -> pd.DataFrame:
    """
    Only the synthetic minority rows SMOTE adds to reach minority/majority == ratio.
    """

    n_fraud = int((y_train == 1).sum())
    current = n_fraud / (len(y_train) - n_fraud)

    if ratio < current:
        raise ValueError(
            f"ratio {ratio} is below the current fraud/legit ratio {current:.4f}; "
            "SMOTE can only add fraud samples, not remove them."
        )

    smote = SMOTE(random_state=seed, sampling_strategy=ratio)  # pyright: ignore[reportArgumentType]
    x_resampled = cast(pd.DataFrame, smote.fit_resample(x_train, y_train)[0])

    return x_resampled.iloc[len(x_train) :].reset_index(drop=True)


def add_synthetic(
    x_train: pd.DataFrame, y_train: pd.Series, synthetic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Original rows first, then the synthetic fraud rows (label 1)
    """

    x = pd.concat([x_train, synthetic], ignore_index=True)
    labels = pd.Series(1, index=range(len(synthetic)), name=y_train.name).astype(y_train.dtype)
    y = pd.concat([y_train, labels], ignore_index=True)

    return x, y
