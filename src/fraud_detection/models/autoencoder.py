import torch
from torch import nn


class AutoEncoder(nn.Module):
    """
    in_dim -> hidden -> latent -> hidden -> in_dim, with optional dropout in the encoder.
    """

    def __init__(self, in_dim: int, hidden: int, latent: int, dropout: float = 0.0) -> None:
        super().__init__()

        encoder: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]

        if dropout > 0:
            encoder.append(nn.Dropout(dropout))

        encoder.append(nn.Linear(hidden, latent))

        self.encoder = nn.Sequential(*encoder)

        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
