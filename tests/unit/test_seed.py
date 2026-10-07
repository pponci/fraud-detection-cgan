import random

import numpy as np

from fraud_detection.utils.seed import set_seed


def test_same_seed_same_numbers() -> None:
    set_seed(7)
    first = (random.random(), float(np.random.rand()))
    set_seed(7)
    second = (random.random(), float(np.random.rand()))

    assert first == second


def test_different_seed_different_numbers() -> None:
    set_seed(7)
    first = random.random()
    set_seed(8)

    assert random.random() != first
