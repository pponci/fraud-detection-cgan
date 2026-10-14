import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class RowDataset(Dataset[torch.Tensor]):
    """
    Each item is one row of rows as a float32 tensor.
    """

    def __init__(self, rows: pd.DataFrame) -> None:
        self.features = torch.as_tensor(rows.to_numpy(dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]


class ConditionedRowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Each item is (row, condition): a row as float32 plus the condition at the same position.
    """

    def __init__(self, rows: pd.DataFrame, condition: NDArray[np.float32]) -> None:
        if len(rows) != len(condition):
            raise ValueError(f"{len(rows)} rows but {len(condition)} conditions")

        self.features = torch.as_tensor(rows.to_numpy(dtype=np.float32, copy=True))
        self.condition = torch.as_tensor(np.array(condition, dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.condition[index]
