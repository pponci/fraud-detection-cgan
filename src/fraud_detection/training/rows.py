import numpy as np
import pandas as pd
import torch
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
