# pyright: reportUnknownMemberType=false

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from fraud_detection.config import GanModelConfig, GanTrainingConfig
from fraud_detection.models.gan import Critic, Generator
from fraud_detection.utils.seed import set_seed


class FraudDataset(Dataset[torch.Tensor]):
    """
    The real fraud rows, as float32 tensors (the GAN never sees legitimate transactions).
    """

    def __init__(self, fraud: pd.DataFrame) -> None:
        self.features = torch.as_tensor(fraud.to_numpy(dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]


@dataclass
class TrainedGan:
    generator: Generator
    critic: Critic
    g_optimizer: optim.Optimizer
    d_optimizer: optim.Optimizer
    noise_dim: int
    data_dim: int
    d_losses: list[float] = field(default_factory=list[float])
    g_losses: list[float] = field(default_factory=list[float])


def gradient_penalty(
    critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    WGAN-GP: push the critic's gradient norm towards 1 on random real/fake interpolations.
    """

    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = cast(torch.Tensor, gradients.norm(2, dim=1))

    return ((gradient_norm - 1) ** 2).mean()


def _bce_step(
    real: torch.Tensor,
    trained: TrainedGan,
    cfg: GanTrainingConfig,
    device: torch.device,
) -> tuple[float, float]:
    """
    Standard GAN step: critic on real then fake (smoothed labels), then generator.
    """

    g, d = trained.generator, trained.critic
    bce = nn.BCEWithLogitsLoss()
    real_label, fake_label = cfg.label_smoothing
    batch_size = real.size(0)

    d.zero_grad()
    real_in = real + cfg.real_noise_std * torch.randn_like(real) if cfg.real_noise_std > 0 else real
    label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

    error_d_real = bce(d(real_in).squeeze(), label)
    error_d_real.backward()

    noise = torch.randn(batch_size, trained.noise_dim).to(device)
    fake = g(noise)
    label.fill_(fake_label)

    error_d_fake = bce(d(fake.detach()).squeeze(), label)
    error_d_fake.backward()

    loss_d = error_d_real + error_d_fake
    trained.d_optimizer.step()

    g.zero_grad()
    label.fill_(1.0)
    error_g = bce(d(fake).squeeze(), label)
    error_g.backward()
    trained.g_optimizer.step()

    return loss_d.item(), error_g.item()


def _wgan_step(
    real: torch.Tensor,
    trained: TrainedGan,
    cfg: GanTrainingConfig,
    device: torch.device,
) -> tuple[float, float]:
    """
    Wasserstein step: n_critic critic updates on this batch, then one generator update.
    """

    g, d = trained.generator, trained.critic
    batch_size = real.size(0)
    real_in = real + cfg.real_noise_std * torch.randn_like(real) if cfg.real_noise_std > 0 else real

    loss_d = torch.zeros(())

    for _ in range(cfg.n_critic):
        d.zero_grad()

        real_validity = d(real_in).mean()
        noise = torch.randn(batch_size, trained.noise_dim, device=device)
        fake = g(noise)
        fake_validity = d(fake.detach()).mean()

        loss_d = -real_validity + fake_validity
        if cfg.lambda_gp > 0:
            loss_d = loss_d + cfg.lambda_gp * gradient_penalty(d, real_in, fake, device)
        loss_d.backward()
        trained.d_optimizer.step()

    g.zero_grad()
    noise = torch.randn(batch_size, trained.noise_dim, device=device)
    fake = g(noise)
    loss_g = -d(fake).mean()
    loss_g.backward()
    trained.g_optimizer.step()

    return loss_d.item(), loss_g.item()


def train_gan(
    fraud: pd.DataFrame,
    model_cfg: GanModelConfig,
    cfg: GanTrainingConfig,
    seed: int,
    device: torch.device,
    log_every: int = 10,
) -> TrainedGan:
    """
    Train on the real fraud rows (predictor columns only). Deterministic for a seed.
    """

    if cfg.objective not in ("bce", "wgan"):
        raise ValueError(f"Unknown objective {cfg.objective!r}; expected 'bce' or 'wgan'")

    set_seed(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        FraudDataset(fraud),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
    )

    if len(loader) == 0:
        raise ValueError(
            f"{len(fraud)} fraud rows cannot fill one batch of {cfg.batch_size}; "
            "lower batch_size or provide more data"
        )

    data_dim = fraud.shape[1]
    generator = Generator(data_dim, model_cfg.noise_dim, model_cfg.generator_hidden).to(device)
    critic = Critic(
        data_dim,
        model_cfg.critic_hidden,
        model_cfg.critic_dropout,
        model_cfg.critic_spectral_norm,
    ).to(device)
    trained = TrainedGan(
        generator=generator,
        critic=critic,
        g_optimizer=optim.Adam(generator.parameters(), lr=cfg.g_lr, betas=cfg.betas),
        d_optimizer=optim.Adam(critic.parameters(), lr=cfg.d_lr, betas=cfg.betas),
        noise_dim=model_cfg.noise_dim,
        data_dim=data_dim,
    )

    step = _bce_step if cfg.objective == "bce" else _wgan_step

    for epoch in range(cfg.epochs):
        generator.train()
        critic.train()

        d_total, g_total = 0.0, 0.0

        for real in loader:
            d_loss, g_loss = step(real.to(device), trained, cfg, device)
            d_total += d_loss
            g_total += g_loss

        trained.d_losses.append(d_total / len(loader))
        trained.g_losses.append(g_total / len(loader))

        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            print(
                f"Epoch {epoch} | D loss: {trained.d_losses[-1]:.6f} "
                f"| G loss: {trained.g_losses[-1]:.6f}"
            )

    return trained


def save_checkpoint(path: str, trained: TrainedGan, epochs: int) -> None:
    torch.save(
        {
            "g_model_state_dict": trained.generator.state_dict(),
            "d_model_state_dict": trained.critic.state_dict(),
            "g_optimizer_state_dict": trained.g_optimizer.state_dict(),
            "d_optimizer_state_dict": trained.d_optimizer.state_dict(),
            "epoch": epochs,
            "noise_dim": trained.noise_dim,
            "data_dim": trained.data_dim,
            "d_train_losses": trained.d_losses,
            "g_train_losses": trained.g_losses,
        },
        path,
    )
