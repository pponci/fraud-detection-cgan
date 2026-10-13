import numpy as np
import pandas as pd
import torch

from fraud_detection.config import AutoencoderModelConfig, AutoencoderTrainingConfig
from fraud_detection.training.autoencoder import train_autoencoder

CPU = torch.device("cpu")
MODEL = AutoencoderModelConfig(hidden=8, latent=3, dropout=0.1)
TRAINING = AutoencoderTrainingConfig(epochs=15, lr=1e-2, batch_size=32)


def make_rows(n: int, seed: int) -> pd.DataFrame:
    """
    Rows lying on a 3-dimensional subspace of 10 features: easy to compress and rebuild.
    """

    rng = np.random.default_rng(seed)
    basis = np.random.default_rng(99).normal(size=(3, 10))

    return pd.DataFrame(rng.normal(size=(n, 3)) @ basis, columns=[f"f{i}" for i in range(10)])


def test_losses_are_recorded_per_epoch_and_improve() -> None:
    trained = train_autoencoder(make_rows(200, 0), make_rows(80, 1), MODEL, TRAINING, 1, CPU)

    assert len(trained.train_losses) == len(trained.val_losses) == TRAINING.epochs
    assert np.isfinite(trained.train_losses + trained.val_losses).all()
    assert trained.train_losses[-1] < trained.train_losses[0]
    assert trained.val_losses[-1] < trained.val_losses[0]


def test_same_seed_gives_identical_models() -> None:
    train, val = make_rows(100, 0), make_rows(40, 1)
    first = train_autoencoder(train, val, MODEL, TRAINING, 3, CPU)
    second = train_autoencoder(train, val, MODEL, TRAINING, 3, CPU)
    other = train_autoencoder(train, val, MODEL, TRAINING, 4, CPU)

    first_state, second_state = first.model.state_dict(), second.model.state_dict()

    assert all(torch.equal(first_state[k], second_state[k]) for k in first_state)
    assert first.train_losses == second.train_losses
    assert not torch.equal(
        first_state["encoder.0.weight"], other.model.state_dict()["encoder.0.weight"]
    )


def test_in_dim_follows_the_data() -> None:
    trained = train_autoencoder(make_rows(64, 0), make_rows(32, 1), MODEL, TRAINING, 1, CPU)

    assert trained.in_dim == 10
