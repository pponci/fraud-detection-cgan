from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from fraud_detection.config import AutoencoderExperimentConfig
from fraud_detection.evaluation.anomaly import AnomalyScorer, error_summary, reconstruction_errors
from fraud_detection.models.autoencoder import AutoEncoder
from fraud_detection.training.autoencoder import save_checkpoint, train_autoencoder


@dataclass
class AutoencoderRunResult:
    summary: pd.DataFrame
    history: pd.DataFrame
    scores: dict[str, pd.DataFrame]
    scorer: AnomalyScorer


def load_autoencoder(
    path: Path, cfg: AutoencoderExperimentConfig, device: torch.device
) -> tuple[AutoEncoder, pd.DataFrame]:
    """
    Load an autoencoder (and its loss history) from a checkpoint.
    """

    checkpoint = torch.load(path, map_location=device)

    model = AutoEncoder(
        checkpoint["in_dim"], cfg.model.hidden, cfg.model.latent, cfg.model.dropout
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    history = pd.DataFrame(
        {"train_loss": checkpoint["train_losses"], "val_loss": checkpoint["val_losses"]}
    )
    history.index.name = "epoch"

    return model, history


def run_autoencoder_experiment(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    cfg: AutoencoderExperimentConfig,
    seed: int,
    device: torch.device,
    checkpoint: Path | None = None,
) -> AutoencoderRunResult:
    """
    Train on the legitimate rows of train (validated on those of val), then compute
    reconstruction errors and anomaly scores for all three splits.
    """

    x = {"train": train.drop(columns=target), "val": val.drop(columns=target)}
    y = {"train": train[target], "val": val[target]}
    x_test = test.drop(columns=target)

    if checkpoint is not None and checkpoint.exists():
        model, history = load_autoencoder(checkpoint, cfg, device)

    else:
        trained = train_autoencoder(
            x["train"][y["train"] == 0],
            x["val"][y["val"] == 0],
            cfg.model,
            cfg.training,
            seed,
            device,
        )
        model = trained.model
        history = pd.DataFrame({"train_loss": trained.train_losses, "val_loss": trained.val_losses})
        history.index.name = "epoch"

        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(str(checkpoint), trained, cfg.training.epochs)

    batch_size = cfg.training.batch_size
    errors = {
        "train": reconstruction_errors(model, x["train"], device, batch_size),
        "val": reconstruction_errors(model, x["val"], device, batch_size),
        "test": reconstruction_errors(model, x_test, device, batch_size),
    }
    summary = pd.DataFrame(
        [error_summary(errors[name], y[name], name) for name in ("train", "val")]
    )

    scorer = AnomalyScorer().fit(errors["train"])
    scores = {name: scorer.transform(err) for name, err in errors.items()}

    return AutoencoderRunResult(summary=summary, history=history, scores=scores, scorer=scorer)
