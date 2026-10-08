from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """
    Noise vector -> one synthetic (scaled) transaction. Linear output, no final activation.
    """

    def __init__(self, data_dim: int, noise_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        width = noise_dim

        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.LeakyReLU(0.2)]
            width = size

        layers.append(nn.Linear(width, data_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Critic(nn.Module):
    """
    Transaction -> one unbounded score (a logit for BCE training, a critic value for WGAN).
    """

    def __init__(
        self,
        data_dim: int,
        hidden: Sequence[int],
        dropout: float = 0.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        def linear(n_in: int, n_out: int) -> nn.Module:
            layer = nn.Linear(n_in, n_out)
            return spectral_norm(layer) if use_spectral_norm else layer

        layers: list[nn.Module] = []
        width = data_dim

        for size in hidden:
            layers += [linear(width, size), nn.LeakyReLU(0.2)]

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            width = size

        layers.append(linear(width, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
