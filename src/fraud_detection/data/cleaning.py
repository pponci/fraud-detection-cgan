import pandas as pd


def missing_summary(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """
    Missing counts per column or per row.

    Sorted by descending missing count. The sort order is deliberate: selecting with the sorted
    labels reorders columns (and train rows).
    """

    if axis == "cols":
        counts, total = df.isna().sum(), len(df)

    elif axis == "rows":
        counts, total = df.isna().sum(axis=1), len(df.columns)

    else:
        raise ValueError("axis must be 'cols' or 'rows'")

    out = pd.DataFrame({"name": counts.index, "n_missing": counts.to_numpy()})
    out = out.sort_values(by="n_missing", ascending=False).reset_index(drop=True)
    out["p_missing"] = [round(int(n) / total, 2) for n in out["n_missing"]]

    return out


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make one-hot column names safe for downstream libraries.
    """

    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"[\[\]{}\"\'\\:;,\s]", "_", regex=True)
        .str.replace(r"-", "_minus_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return df
