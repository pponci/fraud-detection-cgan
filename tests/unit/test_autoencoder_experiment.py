from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud_detection.config import (
    AutoencoderExperimentConfig,
    AutoencoderModelConfig,
    AutoencoderTrainingConfig,
)
from fraud_detection.training.autoencoder_experiment import (
    load_autoencoder,
    run_autoencoder_experiment,
)

CPU = torch.device("cpu")
CONFIG = AutoencoderExperimentConfig(
    name="tiny",
    dataset="synthetic",
    seeds=[1],
    model=AutoencoderModelConfig(hidden=8, latent=3, dropout=0.1),
    training=AutoencoderTrainingConfig(epochs=2, lr=1e-3, batch_size=64),
)


def test_runs_end_to_end_and_reuses_the_checkpoint(
    toy_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = toy_split
    test = val.iloc[:300]
    checkpoint = tmp_path / "ae.pt"

    first = run_autoencoder_experiment(train, val, test, "isFraud", CONFIG, 1, CPU, checkpoint)

    assert checkpoint.exists()
    assert first.summary["dataset"].tolist() == ["train", "val"]
    assert {"n_mean", "f_mean", "n_median", "f_median", "sep_rate"} <= set(first.summary.columns)
    assert len(first.history) == 2 and list(first.history.columns) == ["train_loss", "val_loss"]
    assert {name: len(s) for name, s in first.scores.items()} == {
        "train": len(train),
        "val": len(val),
        "test": 300,
    }
    assert all(list(s.columns) == ["ae_score"] for s in first.scores.values())
    assert np.isfinite(first.scores["train"].to_numpy()).all()

    second = run_autoencoder_experiment(train, val, test, "isFraud", CONFIG, 1, CPU, checkpoint)

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.scores["val"], second.scores["val"])
    pd.testing.assert_frame_equal(first.history, second.history)


def test_checkpoint_keeps_the_documented_layout(
    toy_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = toy_split
    checkpoint = tmp_path / "ae.pt"
    run_autoencoder_experiment(train, val, val, "isFraud", CONFIG, 1, CPU, checkpoint)

    saved = torch.load(checkpoint, map_location=CPU)

    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "in_dim",
        "train_losses",
        "val_losses",
    } <= set(saved)
    assert "state" in saved["optimizer_state_dict"]

    model, history = load_autoencoder(checkpoint, CONFIG, CPU)

    assert model(torch.randn(2, saved["in_dim"])).shape == (2, saved["in_dim"])
    assert len(history) == 2
