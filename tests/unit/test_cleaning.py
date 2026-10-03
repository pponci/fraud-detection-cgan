import pandas as pd

from fraud_detection.data.cleaning import clean_column_names, missing_summary


def test_missing_summary_sorted_by_descending_missing() -> None:
    df = pd.DataFrame({"a": [None, None, 1, 2], "b": [1, 2, 3, 4], "c": [None, 1, 2, 3]})
    out = missing_summary(df, "cols")

    assert out["name"].tolist() == ["a", "c", "b"]
    assert out["p_missing"].tolist() == [0.5, 0.25, 0.0]


def test_missing_summary_rows_uses_column_count_as_denominator() -> None:
    df = pd.DataFrame({"a": [None, 1], "b": [None, 2], "c": [3, 4], "d": [None, 5]})
    out = missing_summary(df, "rows")

    assert out["name"].tolist() == [0, 1]
    assert out["p_missing"].tolist() == [0.75, 0.0]


def test_p_missing_uses_python_rounding_not_numpy() -> None:
    df = pd.DataFrame({"a": [None] + [1] * 39})

    assert missing_summary(df, "cols")["p_missing"].tolist() == [0.03]


def test_clean_column_names() -> None:
    df = pd.DataFrame(columns=["card4_american express", "a-b", "x[1]", "M4_unknown ", "ok"])

    assert clean_column_names(df).columns.tolist() == [
        "card4_american_express",
        "a_minus_b",
        "x_1",
        "M4_unknown",
        "ok",
    ]
