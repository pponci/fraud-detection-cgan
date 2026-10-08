from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud_detection.config import (
    GanExperimentConfig,
    GanModelConfig,
    GanTrainingConfig,
)
from fraud_detection.training.gan_experiment import load_generator, run_gan_experiment

CPU = torch.device("cpu")
CONFIG = GanExperimentConfig(
    name="tiny",
    dataset="synthetic",
    seeds=[1],
    model=GanModelConfig(noise_dim=8, generator_hidden=[16], critic_hidden=[16]),
    training=GanTrainingConfig(
        objective="wgan", epochs=2, g_lr=1e-3, d_lr=1e-3, batch_size=32, lambda_gp=1.0
    ),
    ratios=[0.2, 0.3],
    classifier_params={"n_estimators": 10, "n_jobs": 1, "verbose": -1},
)


def test_runs_end_to_end_and_reuses_the_checkpoint(
    toy_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = toy_split
    checkpoint = tmp_path / "gan.pt"

    first = run_gan_experiment(train, val, "isFraud", CONFIG, 1, CPU, checkpoint)

    assert checkpoint.exists()
    assert len(first.metrics) == 4
    assert set(first.metrics["method"]) == {"gan"}
    assert first.synthetic_quality["ratio"].tolist() == [0.2, 0.3]
    assert len(first.history) == 2
    assert np.isfinite(first.history.to_numpy()).all()

    second = run_gan_experiment(train, val, "isFraud", CONFIG, 1, CPU, checkpoint)

    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.history, second.history)


def test_checkpoint_keeps_the_documented_layout(
    toy_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = toy_split
    checkpoint = tmp_path / "gan.pt"
    run_gan_experiment(train, val, "isFraud", replace(CONFIG, ratios=[0.2]), 1, CPU, checkpoint)

    saved = torch.load(checkpoint, map_location=CPU)

    assert {
        "g_model_state_dict",
        "d_model_state_dict",
        "g_optimizer_state_dict",
        "d_optimizer_state_dict",
        "epoch",
        "noise_dim",
        "data_dim",
        "d_train_losses",
        "g_train_losses",
    } <= set(saved)

    generator, history = load_generator(checkpoint, CONFIG, CPU)

    assert generator(torch.randn(2, 8)).shape == (2, saved["data_dim"])
    assert len(history) == 2
