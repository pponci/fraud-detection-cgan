import pandas as pd


def temporal_split(
    df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-sorted frame by position.
    """

    n = len(df)
    cutoff_train = int(n * (1 - val_size - test_size))
    cutoff_val = int(n * (1 - test_size))

    train = df.iloc[:cutoff_train].copy().reset_index(drop=True)
    val = df.iloc[cutoff_train:cutoff_val].copy().reset_index(drop=True)
    test = df.iloc[cutoff_val:].copy().reset_index(drop=True)

    return train, val, test
