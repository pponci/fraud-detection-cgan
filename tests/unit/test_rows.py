import numpy as np
import pandas as pd
import pytest

from fraud_detection.training.rows import ConditionedRowDataset, RowDataset


def test_row_dataset_yields_float32_rows() -> None:
    dataset = RowDataset(pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}))

    assert len(dataset) == 3
    assert dataset[1].tolist() == [2.0, 5.0] and str(dataset[1].dtype) == "torch.float32"


def test_conditioned_dataset_pairs_row_i_with_condition_i() -> None:
    rows = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
    condition = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    dataset = ConditionedRowDataset(rows, condition)
    x, cond = dataset[2]

    assert x.tolist() == [30.0] and cond.tolist() == [3.0]


def test_conditioned_dataset_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="3 rows but 2 conditions"):
        ConditionedRowDataset(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), np.zeros((2, 1), np.float32))
