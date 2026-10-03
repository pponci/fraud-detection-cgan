import pandas as pd

from fraud_detection.data.splitting import temporal_split


def test_sizes_and_order() -> None:
    df = pd.DataFrame({"TransactionDT": range(1000), "x": range(1000)})
    train, val, test = temporal_split(df, val_size=0.2, test_size=0.1)

    assert (len(train), len(val), len(test)) == (700, 200, 100)
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()


def test_no_overlap_and_full_coverage() -> None:
    df = pd.DataFrame({"TransactionDT": range(333)})
    parts = temporal_split(df)

    assert sum(map(len, parts)) == 333
    assert pd.concat(parts)["TransactionDT"].is_unique
