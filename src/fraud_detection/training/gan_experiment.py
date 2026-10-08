from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from fraud_detection.augmentation.gan import generate_gan_samples, n_synthetic_for_share
from fraud_detection.config import GanExperimentConfig
from fraud_detection.models.gan import Generator
from fraud_detection.training.gan import save_checkpoint, train_gan
from fraud_detection.training.sweep import evaluate_synthetic_sweep


@dataclass
class GanRunResult:
    metrics: pd.DataFrame
    synthetic_quality: pd.DataFrame
    history: pd.DataFrame


def load_generator(
    path: Path, cfg: GanExperimentConfig, device: torch.device
) -> tuple[Generator, pd.DataFrame]:
    """
    Load a generator (and its loss history) from a checkpoint written by save_checkpoint.
    """

    checkpoint = torch.load(path, map_location=device)
    generator = Generator(
        checkpoint["data_dim"], checkpoint["noise_dim"], cfg.model.generator_hidden
    )
    generator.load_state_dict(checkpoint["g_model_state_dict"])
    generator.to(device)
    history = pd.DataFrame(
        {"d_loss": checkpoint["d_train_losses"], "g_loss": checkpoint["g_train_losses"]}
    )
    history.index.name = "epoch"

    return generator, history


def run_gan_experiment(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    cfg: GanExperimentConfig,
    seed: int,
    device: torch.device,
    checkpoint: Path | None = None,
) -> GanRunResult:
    """
    Train a GAN on the real fraud rows of train and evaluate it on the ratio sweep.
    """

    x_train, y_train = train.drop(columns=target), train[target]

    if checkpoint is not None and checkpoint.exists():
        generator, history = load_generator(checkpoint, cfg, device)

    else:
        trained = train_gan(x_train[y_train == 1], cfg.model, cfg.training, seed, device)
        generator = trained.generator
        history = pd.DataFrame({"d_loss": trained.d_losses, "g_loss": trained.g_losses})
        history.index.name = "epoch"

        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(str(checkpoint), trained, cfg.training.epochs)

    def make_synthetic(ratio: float) -> pd.DataFrame:
        return generate_gan_samples(
            generator,
            n_rows=n_synthetic_for_share(y_train, ratio),
            noise_dim=cfg.model.noise_dim,
            columns=x_train.columns,
            seed=seed,
            device=device,
        )

    metrics, quality = evaluate_synthetic_sweep(
        "gan", cfg.ratios, make_synthetic, train, val, target, seed, cfg.classifier_params
    )

    return GanRunResult(metrics=metrics, synthetic_quality=quality, history=history)
