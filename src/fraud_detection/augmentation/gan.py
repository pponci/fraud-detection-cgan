import numpy as np
import pandas as pd
import torch

from fraud_detection.models.gan import Generator


def n_synthetic_for_share(y_train: pd.Series, target_share: float) -> int:
    """
    Rows to add so that the fraud count equals target_share x the ORIGINAL row count.
    """

    actual_share = float((y_train == 1).sum()) / len(y_train)
    n_rows = round((target_share - actual_share) * len(y_train))

    if n_rows < 0:
        raise ValueError(
            f"target share {target_share} is below the current fraud share {actual_share:.4f}"
        )

    return n_rows


def generate_gan_samples(
    generator: Generator,
    n_rows: int,
    noise_dim: int,
    columns: pd.Index,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Draw n_rows synthetic samples. The same seed and size always give the same samples.
    """

    generator.eval()
    noise_generator = torch.Generator()
    noise_generator.manual_seed(seed)
    noise = torch.randn(n_rows, noise_dim, generator=noise_generator).to(device)

    with torch.no_grad():
        fake = generator(noise).cpu().numpy()

    return pd.DataFrame(fake.astype(np.float64), columns=columns)
