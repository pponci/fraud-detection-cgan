from dataclasses import dataclass, field

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from fraud_detection.config import AutoencoderModelConfig, AutoencoderTrainingConfig
from fraud_detection.models.autoencoder import AutoEncoder
from fraud_detection.training.rows import RowDataset
from fraud_detection.utils.seed import set_seed


@dataclass
class TrainedAutoencoder:
    model: AutoEncoder
    optimizer: optim.Optimizer
    in_dim: int
    train_losses: list[float] = field(default_factory=list[float])
    val_losses: list[float] = field(default_factory=list[float])


def train_autoencoder(
    train_legit: pd.DataFrame,
    val_legit: pd.DataFrame,
    model_cfg: AutoencoderModelConfig,
    cfg: AutoencoderTrainingConfig,
    seed: int,
    device: torch.device,
) -> TrainedAutoencoder:
    """
    Fit on legitimate rows only, so fraud shows up as poorly reconstructed.
    """

    set_seed(seed)
    train_loader = DataLoader(RowDataset(train_legit), batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(RowDataset(val_legit), batch_size=cfg.batch_size, shuffle=False)

    in_dim = train_legit.shape[1]
    model = AutoEncoder(in_dim, model_cfg.hidden, model_cfg.latent, model_cfg.dropout).to(device)
    trained = TrainedAutoencoder(
        model=model, optimizer=optim.Adam(model.parameters(), lr=cfg.lr), in_dim=in_dim
    )
    loss_fn = nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for real in train_loader:
            real = real.to(device)
            trained.optimizer.zero_grad()
            loss = loss_fn(model(real), real)
            loss.backward()
            trained.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        trained.train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for real in val_loader:
                real = real.to(device)
                val_loss += loss_fn(model(real), real).item()

        val_loss /= len(val_loader)
        trained.val_losses.append(val_loss)

        print(f"Epoch {epoch} | train loss : {train_loss:.6f} | val loss : {val_loss:.6f}")

    return trained


def save_checkpoint(path: str, trained: TrainedAutoencoder, epochs: int) -> None:
    torch.save(
        {
            "model_state_dict": trained.model.state_dict(),
            "optimizer_state_dict": trained.optimizer.state_dict(),
            "epoch": epochs,
            "in_dim": trained.in_dim,
            "train_losses": trained.train_losses,
            "val_losses": trained.val_losses,
        },
        path,
    )
