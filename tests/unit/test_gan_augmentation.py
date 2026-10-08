import numpy as np
import pandas as pd
import pytest
import torch

from fraud_detection.augmentation.gan import generate_gan_samples, n_synthetic_for_share
from fraud_detection.models.gan import Generator

CPU = torch.device("cpu")


def test_rows_needed_for_a_target_share() -> None:
    y = pd.Series([1] * 100 + [0] * 900)

    assert n_synthetic_for_share(y, 0.2) == 100
    assert n_synthetic_for_share(y, 0.1) == 0


def test_target_below_current_share_is_an_error() -> None:
    y = pd.Series([1] * 100 + [0] * 900)

    with pytest.raises(ValueError, match="below the current fraud share"):
        n_synthetic_for_share(y, 0.05)


def test_share_is_defined_on_original_rows_unlike_smote_ratio() -> None:
    y = pd.Series([1] * 100 + [0] * 900)

    assert n_synthetic_for_share(y, 0.5) == 400


@pytest.fixture
def generator() -> Generator:
    torch.manual_seed(0)

    return Generator(data_dim=5, noise_dim=4, hidden=[8])


def test_samples_have_the_right_shape_columns_and_dtype(generator: Generator) -> None:
    columns = pd.Index(list("abcde"))
    samples = generate_gan_samples(generator, 12, 4, columns, seed=1, device=CPU)

    assert samples.shape == (12, 5)
    assert samples.columns.tolist() == list("abcde")
    assert (samples.dtypes == np.float64).all()


def test_same_seed_and_size_give_the_same_samples(generator: Generator) -> None:
    columns = pd.Index(list("abcde"))
    first = generate_gan_samples(generator, 20, 4, columns, seed=1, device=CPU)
    second = generate_gan_samples(generator, 20, 4, columns, seed=1, device=CPU)
    other = generate_gan_samples(generator, 20, 4, columns, seed=2, device=CPU)
    pd.testing.assert_frame_equal(first, second)

    assert not first.equals(other)
