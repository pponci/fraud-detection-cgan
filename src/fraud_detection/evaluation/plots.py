# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from pathlib import Path

import numpy as np
import sklearn.metrics as skm
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

PALETTE = ["dimgray", "darkorange", "steelblue", "forestgreen", "firebrick", "purple"]


def plot_pr_curves(
    y_true: ArrayLike,
    probas: dict[str, NDArray[np.float64]],
    pr_aucs: dict[str, float],
    colors: dict[str, str],
    title: str,
    path: Path,
) -> None:
    """
    Precision-recall curves of several methods on the same labels.
    """

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot()

    for label, proba in probas.items():
        precision, recall, _ = skm.precision_recall_curve(y_true, proba)
        ax.plot(
            recall * 100,
            precision * 100,
            label=f"{label} (PR-AUC = {pr_aucs[label]:.4f})",
            color=colors[label],
        )

    ax.set_xlabel("Recall (fraud caught) %")
    ax.set_ylabel("Precision (alert accuracy) %")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_false_alerts_saved(
    recall_targets: list[float],
    saved: dict[str, list[int]],
    colors: dict[str, str],
    title: str,
    path: Path,
) -> None:
    """
    False alerts avoided relative to the baseline, for each recall the bank might require.
    """

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot()

    for label, values in saved.items():
        ax.plot(recall_targets, values, label=label, color=colors[label], marker="o", markersize=4)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("False alerts saved\n(compared with the baseline)")
    ax.set_title(title)
    ax.set_xticks(list(np.arange(0.0, 1.05, 0.1)))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    _save(fig, path)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
