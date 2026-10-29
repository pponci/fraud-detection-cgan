from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.evaluation.experiment_tables import describe, render_experiments


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """A small configs/ and results/ tree: 2 autoencoders, 2 GANs, 1 cGAN, the SMOTE sweep."""
    configs, results = tmp_path / "configs", tmp_path / "results"
    dev, final = configs / "development", configs / "final" / "toy"
    write(dev / "ae/ae_01.yaml", "# Autoencoder iteration 1: small, no dropout\nname: ae_01\n")
    write(dev / "ae/ae_02.yaml", "# Autoencoder iteration 2: dropout\nname: ae_02\n")
    write(dev / "gan/gan_01.yaml", "# GAN iteration 1: standard GAN\nname: gan_01\n")
    write(dev / "gan/gan_02.yaml", "# GAN iteration 2: Wasserstein loss\nname: gan_02\n")
    write(
        dev / "cgan/cgan_01.yaml",
        "# cGAN iteration 1: plus the condition\nname: cgan_01\ncondition:\n  scores_from: ae_02\n",
    )
    write(final / "gan.yaml", "name: gan\nsource: gan_02\n")
    write(final / "ae_cgan.yaml", "name: ae_cgan\nsource: cgan_01\n")
    write(final / "baseline.yaml", "name: baseline\n")

    def sweep(name: str, method: str, best: float, worse: float) -> None:
        folder = results / "toy" / name / "seed_11"
        folder.mkdir(parents=True)
        rows = [
            (method, 0.05, "val", worse), (method, 0.1, "val", best), (method, 0.1, "train", 0.99),
        ]  # fmt: skip
        pd.DataFrame(rows, columns=["method", "ratio", "dataset", "pr_auc"]).to_csv(
            folder / "metrics.csv", index=False
        )

    sweep("gan_02", "gan", 0.5187, 0.5000)
    sweep("cgan_01", "cgan", 0.5185, 0.5100)
    smote = results / "toy" / "smote_sweep" / "seed_11"
    smote.mkdir(parents=True)
    pd.DataFrame(
        [("base", float("nan"), "val", 0.5076), ("smote", 0.05, "val", 0.5178)],
        columns=["method", "ratio", "dataset", "pr_auc"],
    ).to_csv(smote / "metrics.csv", index=False)
    ae = results / "toy" / "ae_02" / "seed_11"
    ae.mkdir(parents=True)
    pd.DataFrame([("train", 3.7673), ("val", 2.5411)], columns=["dataset", "sep_rate"]).to_csv(
        ae / "metrics.csv", index=False
    )

    return configs, results


def test_describe_reads_the_text_after_the_colon(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("# GAN iteration 3: Wasserstein loss, 5 critic steps\nname: x\n")

    assert describe(path) == "Wasserstein loss, 5 critic steps"

    path.write_text("# No colon here\n")

    assert describe(path) == "No colon here"


def test_every_experiment_appears_and_unrun_ones_say_so(roots: tuple[Path, Path]) -> None:
    configs, results = roots
    text = render_experiments("toy", 11, results, configs)

    for name in ("ae_01", "ae_02", "gan_01", "gan_02", "cgan_01"):
        assert name in text

    assert "| ae_01 | small, no dropout | not run | not run |" in text
    assert "| gan_01 | standard GAN | not run | not run |" in text


def test_the_numbers_come_from_the_results(roots: tuple[Path, Path]) -> None:
    configs, results = roots
    text = render_experiments("toy", 11, results, configs)

    assert "| 3.767 | 2.541 |" in text
    assert "| 0.1 | 0.5187 |" in text
    assert "| baseline (no oversampling) | 0.5076 |" in text
    assert "| 0.05 | 0.5178 |" in text


def test_the_experiments_behind_the_final_comparison_are_marked(
    roots: tuple[Path, Path],
) -> None:
    configs, results = roots
    text = render_experiments("toy", 11, results, configs)

    assert "**gan_02** (used)" in text and "| gan_01 |" in text
    assert "**cgan_01** (used)" in text
    assert "**ae_02** (used)" in text
    assert "**ae_01**" not in text


def test_the_document_says_it_is_generated(roots: tuple[Path, Path]) -> None:
    configs, results = roots
    text = render_experiments("toy", 11, results, configs)

    assert text.startswith("# Experiments") and "generated" in text
    assert "seed 11 on `toy`" in text


def test_a_config_with_only_comments_does_not_break_the_document(roots: tuple[Path, Path]) -> None:
    configs, results = roots
    write(configs / "development/gan/gan_03.yaml", "# GAN iteration 3: comments only\n")
    text = render_experiments("toy", 11, results, configs)

    assert "| gan_03 | comments only | not run | not run |" in text
