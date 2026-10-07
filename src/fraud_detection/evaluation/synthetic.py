# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as sps
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


def wasserstein_dist(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """
    Mean over features of the 1-D Wasserstein distance
    """

    dists = [float(sps.wasserstein_distance(real[col], synth[col])) for col in real.columns]

    return float(np.mean(dists))


def corr_distance(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """
    Mean absolute difference of the correlation matrices
    """

    diff = np.abs(real.corr().to_numpy() - synth.corr().to_numpy())

    return float(np.nanmean(diff))


def nn_distance(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[float, float]:
    """
    Mean distance from each synthetic sample to its nearest real sample.
    """

    dist_to_real = cast(
        NDArray[np.float64], NearestNeighbors(n_neighbors=1).fit(real).kneighbors(synth)[0]
    )

    dist_to_fake = cast(
        NDArray[np.float64], NearestNeighbors(n_neighbors=1).fit(synth).kneighbors(real)[0]
    )

    return float(dist_to_real.mean()), float(dist_to_fake.mean())


def synthetic_quality(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:

    dist_to_real, dist_to_fake = nn_distance(real, synth)

    return {
        "Wasserstein": wasserstein_dist(real, synth),
        "Corr_diff": corr_distance(real, synth),
        "NN_real_to_fake": dist_to_real,
        "NN_fake_to_real": dist_to_fake,
    }
