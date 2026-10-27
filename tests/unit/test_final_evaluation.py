from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from lightgbm import LGBMClassifier

import fraud_detection.training.final_evaluation as final_module
from fraud_detection.config import FinalMethodConfig
from fraud_detection.evaluation.report import write_report
from fraud_detection.models.gan import Generator
from fraud_detection.training.final_evaluation import (
    FinalData,
    FinalResult,
    prepare_method,
    run_final_evaluation,
    select_ratio,
)

CPU = torch.device("cpu")
PARAMS: dict[str, Any] = {"n_estimators": 15, "n_jobs": 1, "verbose": -1}


def cfg(
    name: str, method: str, source: str | None = None, ratio: float | None = None
) -> FinalMethodConfig:
    return FinalMethodConfig(name, name.upper(), "toy", method, [1], PARAMS, source, ratio)


BASELINE = cfg("base", "baseline")
SMOTE = cfg("smote", "smote", "smote_sweep", 0.3)
GAN = cfg("gan", "gan", "gan_08", 0.2)
CGAN = cfg("cgan", "cgan", "cgan_04", 0.25)


def write_generator(path: Path, data_dim: int, cond_dim: int) -> None:
    """
    A random generator with the architecture of the final configs, in the checkpoint format.
    """

    torch.manual_seed(0)
    generator = Generator(data_dim, 128, [512, 512, 512, 256], cond_dim)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "g_model_state_dict": generator.state_dict(),
            "noise_dim": 128,
            "data_dim": data_dim,
            "cond_dim": cond_dim,
            "d_train_losses": [0.0],
            "g_train_losses": [0.0],
        },
        path,
    )


def make_scores(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=len(df)) + 1.5 * df["isFraud"].to_numpy()

    return pd.DataFrame({"ae_score": values.astype(np.float32)})


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Empty results/ and artifacts/ folders that the pipeline reads from and writes to.
    """

    monkeypatch.setattr(final_module, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(final_module, "ARTIFACTS_DIR", tmp_path / "artifacts")

    return tmp_path


@pytest.fixture
def toy_data(toy_split: tuple[pd.DataFrame, pd.DataFrame], artifacts: Path) -> FinalData:
    train, val = toy_split
    test = val.iloc[:300].reset_index(drop=True)
    data_dim = train.shape[1] - 1
    write_generator(artifacts / "artifacts/toy/gan_08/seed_1/gan.pt", data_dim, 0)
    write_generator(artifacts / "artifacts/toy/cgan_04/seed_1/gan.pt", data_dim, 1)
    scores = {
        "train": make_scores(train, 1),
        "val": make_scores(val, 2),
        "test": make_scores(test, 3),
    }

    return FinalData(train, val, test, "isFraud", scores)


def write_sweep(artifacts: Path, rows: list[tuple[str, float, str, float]]) -> None:
    folder = artifacts / "results/toy/smote_sweep/seed_1"
    folder.mkdir(parents=True)
    table = pd.DataFrame(rows, columns=["method", "ratio", "dataset", "pr_auc"])
    table.to_csv(folder / "metrics.csv", index=False)


def test_select_ratio_takes_the_best_validation_pr_auc(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "method": ["base", "smote", "smote", "smote", "smote", "gan"],
            "ratio": [np.nan, 0.05, 0.075, 0.05, 0.075, 0.1],
            "dataset": ["val", "val", "val", "train", "train", "val"],
            "pr_auc": [0.9, 0.51, 0.50, 0.99, 0.99, 0.80],
        }
    )
    path = tmp_path / "metrics.csv"
    table.to_csv(path, index=False)

    assert select_ratio(path, "smote") == (0.05, 0.51)


def test_select_ratio_prefers_the_first_on_ties_and_rejects_missing_methods(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {"method": ["gan"] * 2, "ratio": [0.1, 0.2], "dataset": ["val"] * 2, "pr_auc": [0.5, 0.5]}
    ).to_csv(path, index=False)

    assert select_ratio(path, "gan")[0] == 0.1
    with pytest.raises(ValueError, match="no validation rows"):
        select_ratio(path, "cgan")


def test_an_auto_ratio_is_read_from_the_development_results(artifacts: Path) -> None:
    write_sweep(
        artifacts,
        [
            ("smote", 0.05, "val", 0.52),
            ("smote", 0.075, "val", 0.51),
            ("base", float("nan"), "val", 0.9),
        ],
    )
    prepared = prepare_method(cfg("smote", "smote", "smote_sweep"), 1, CPU)

    assert prepared.ratio == 0.05
    assert "smote_sweep" in prepared.selection and "0.5200" in prepared.selection


def test_missing_development_results_and_generators_give_clear_errors(artifacts: Path) -> None:

    with pytest.raises(FileNotFoundError, match="run the smote_sweep experiment first"):
        prepare_method(cfg("smote", "smote", "smote_sweep"), 1, CPU)
    with pytest.raises(FileNotFoundError, match="train gan_08 first"):
        prepare_method(GAN, 1, CPU)


def test_a_fixed_ratio_needs_no_development_results(artifacts: Path) -> None:
    prepared = prepare_method(SMOTE, 1, CPU)

    assert prepared.ratio == 0.3 and prepared.selection == "fixed in the config"


def test_the_baseline_has_no_ratio() -> None:
    assert prepare_method(BASELINE, 1, CPU).ratio is None


def test_the_comparison_needs_a_baseline(toy_data: FinalData) -> None:
    with pytest.raises(ValueError, match="needs a baseline"):
        run_final_evaluation([SMOTE], toy_data, 1, CPU)


def test_a_cgan_needs_the_anomaly_scores(toy_data: FinalData) -> None:
    toy_data.scores = None

    with pytest.raises(ValueError, match="conditional but no scores"):
        run_final_evaluation([BASELINE, CGAN], toy_data, 1, CPU)


@pytest.fixture
def result(
    toy_data: FinalData, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[FinalResult, list[int]]]:
    fit_sizes: list[int] = []
    original = final_module.fit_classifier

    def spy(
        x: pd.DataFrame, y: pd.Series, seed: int, params: dict[str, Any] | None = None
    ) -> LGBMClassifier:
        fit_sizes.append(len(x))
        return original(x, y, seed, params)

    monkeypatch.setattr(final_module, "fit_classifier", spy)

    yield run_final_evaluation([BASELINE, SMOTE, GAN, CGAN], toy_data, 1, CPU), fit_sizes


def test_both_stages_report_every_method(result: tuple[FinalResult, list[int]]) -> None:
    final, _ = result

    for stage, eval_name in ((final.val, "val"), (final.test, "test")):
        assert set(stage.metrics["method"]) == {"BASE", "SMOTE", "GAN", "CGAN"}
        assert set(stage.metrics["dataset"]) == {"train", eval_name}
        assert len(stage.metrics) == 8
        assert "precision_at_95_recall" in stage.metrics.columns

    assert final.selected_ratios["ratio"].tolist() == [0.3, 0.2, 0.25]


def test_synthetic_quality_is_measured_on_validation_only(
    result: tuple[FinalResult, list[int]],
) -> None:
    final, _ = result

    assert set(final.val.quality["method"]) == {"SMOTE", "GAN", "CGAN"}
    assert final.test.quality.empty


def test_the_test_stage_is_fitted_on_train_plus_validation(
    result: tuple[FinalResult, list[int]], toy_data: FinalData
) -> None:
    _, fit_sizes = result
    n_train, n_val = len(toy_data.train), len(toy_data.val)
    baseline_val_fit, baseline_test_fit = fit_sizes[0], fit_sizes[4]

    assert baseline_val_fit == n_train
    assert baseline_test_fit == n_train + n_val
    assert all(size > n_train for size in fit_sizes[1:4])


def test_predictions_and_false_alerts_are_consistent(
    result: tuple[FinalResult, list[int]], toy_data: FinalData
) -> None:
    final, _ = result

    assert all(len(p) == len(toy_data.val) for p in final.val.probabilities.values())
    assert all(len(p) == len(toy_data.test) for p in final.test.probabilities.values())

    baseline = final.false_alerts_test[final.false_alerts_test["method"] == "BASE"]

    assert (baseline["saved_vs_baseline"] == 0).all()
    assert len(final.false_alerts_val) == 4 * 20


def test_the_report_writes_tables_figures_and_predictions(
    result: tuple[FinalResult, list[int]], tmp_path: Path
) -> None:
    final, _ = result
    results_dir, artifacts_dir = tmp_path / "results", tmp_path / "artifacts"
    write_report(final, results_dir, artifacts_dir)

    written = {p.name for p in results_dir.rglob("*") if p.is_file()}

    assert {
        "selected_ratios.csv",
        "val_metrics.csv",
        "test_metrics.csv",
        "synthetic_metrics.csv",
        "false_alerts_val.csv",
        "false_alerts_test.csv",
        "pr_curves_val.png",
        "pr_curves_test.png",
        "false_alerts_val.png",
        "false_alerts_test.png",
    } <= written
    predictions = pd.read_parquet(artifacts_dir / "predictions_test.parquet")

    assert list(predictions.columns) == ["y_true", "BASE", "SMOTE", "GAN", "CGAN"]
