# pyright: reportUnknownMemberType=false

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed python, numpy and torch, ask torch for
    deterministic kernels.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    """
    Automatically sets cuda (gpu) if available
    if not defualts to cpu.
    """

    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"

    return torch.device(device)
