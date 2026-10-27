from pathlib import Path

import numpy as np

from fraud_detection.evaluation.plots import plot_false_alerts_saved, plot_pr_curves

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_pr_curves_are_written_as_a_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    probas = {"A": rng.random(500) * 0.5 + 0.4 * y, "B": rng.random(500)}
    path = tmp_path / "figures" / "pr.png"
    plot_pr_curves(y, probas, {"A": 0.5, "B": 0.1}, {"A": "red", "B": "blue"}, "PR", path)

    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_false_alerts_saved_figure_is_written_as_a_png(tmp_path: Path) -> None:
    targets = [0.05, 0.5, 1.0]
    path = tmp_path / "saved.png"
    saved = {"GAN": [3, -2, 10], "AE-cGAN": [1, 0, 4]}
    plot_false_alerts_saved(targets, saved, {"GAN": "red", "AE-cGAN": "blue"}, "saved", path)

    assert path.read_bytes().startswith(PNG_SIGNATURE)
