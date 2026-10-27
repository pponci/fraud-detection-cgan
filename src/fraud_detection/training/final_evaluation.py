from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from fraud_detection.augmentation.condition import sample_condition_scores
from fraud_detection.augmentation.gan import generate_gan_samples, n_synthetic_for_share
from fraud_detection.augmentation.smote import add_synthetic, smote_synthetic
from fraud_detection.config import (
    ARTIFACTS_DIR,
    RESULTS_DIR,
    FinalMethodConfig,
    GanExperimentConfig,
    find_experiment_config,
    load_gan_config,
)
from fraud_detection.evaluation.ranking import RECALL_TARGETS, false_alerts, ranking_metrics
from fraud_detection.evaluation.synthetic import synthetic_quality
from fraud_detection.models.gan import Generator
from fraud_detection.training.classifier import fit_classifier, predict_scores
from fraud_detection.training.gan_experiment import load_generator


@dataclass
class FinalData:
    """
    The three processed splits and, for a cGAN, the anomaly scores of each split.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target: str
    scores: dict[str, pd.DataFrame] | None = None


@dataclass
class PreparedMethod:
    cfg: FinalMethodConfig
    ratio: float | None
    selection: str
    generator: Generator | None = None
    gan_cfg: GanExperimentConfig | None = None


@dataclass
class StageResult:
    metrics: pd.DataFrame
    probabilities: dict[str, NDArray[np.float64]]
    quality: pd.DataFrame


@dataclass
class FinalResult:
    selected_ratios: pd.DataFrame
    val: StageResult
    test: StageResult
    y_val: NDArray[np.int64]
    y_test: NDArray[np.int64]
    false_alerts_val: pd.DataFrame
    false_alerts_test: pd.DataFrame


def select_ratio(metrics_csv: Path, method: str) -> tuple[float, float]:
    """
    The ratio with the best validation PR-AUC among method's rows, and that PR-AUC.
    """

    table = pd.read_csv(metrics_csv)
    val_rows = table[(table["dataset"] == "val") & (table["method"] == method)]

    if val_rows.empty:
        raise ValueError(f"no validation rows for method {method!r} in {metrics_csv}")

    ratios = val_rows["ratio"].to_numpy(dtype=float)
    pr_aucs = val_rows["pr_auc"].to_numpy(dtype=float)
    best = int(np.argmax(pr_aucs))

    return float(ratios[best]), float(pr_aucs[best])


def prepare_method(cfg: FinalMethodConfig, seed: int, device: torch.device) -> PreparedMethod:
    """
    Resolve the oversampling ratio and load the generator (if the method has one).
    """

    if cfg.method == "baseline":
        return PreparedMethod(cfg, None, "-")

    ratio, selection = cfg.ratio, "fixed in the config"

    if ratio is None:
        assert cfg.source is not None

        results_csv = RESULTS_DIR / cfg.dataset / cfg.source / f"seed_{seed}" / "metrics.csv"

        if not results_csv.exists():
            raise FileNotFoundError(
                f"{results_csv} not found: run the {cfg.source} experiment first, or set a "
                f"fixed `ratio` in the {cfg.name} config"
            )

        ratio, best = select_ratio(results_csv, cfg.method)
        selection = f"best validation PR-AUC in {cfg.source} ({best:.4f})"

    if cfg.method == "smote":
        return PreparedMethod(cfg, ratio, selection)

    assert cfg.source is not None

    gan_cfg = load_gan_config(find_experiment_config(cfg.source))
    checkpoint = ARTIFACTS_DIR / cfg.dataset / cfg.source / f"seed_{seed}" / "gan.pt"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found: train {cfg.source} first with scripts/train_gan.py "
            f"(seed {seed})"
        )
    generator, _ = load_generator(checkpoint, gan_cfg, device)

    return PreparedMethod(cfg, ratio, selection, generator, gan_cfg)


def _training_set(
    method: PreparedMethod,
    x: pd.DataFrame,
    y: pd.Series,
    scores: pd.DataFrame | None,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """
    The data this method's classifier is fitted on, and its synthetic rows (if any).
    """

    if method.ratio is None:
        return x, y, None

    if method.cfg.method == "smote":
        synthetic = smote_synthetic(x, y, method.ratio, seed)

    else:
        assert method.generator is not None and method.gan_cfg is not None

        n_rows = n_synthetic_for_share(y, method.ratio)
        condition = None

        if method.gan_cfg.condition is not None:
            if scores is None:
                raise ValueError(f"{method.cfg.name} is conditional but no scores were given")

            n_bins = method.gan_cfg.condition.n_bins
            condition = sample_condition_scores(scores, y, n_rows, seed, n_bins)

        synthetic = generate_gan_samples(
            method.generator,
            n_rows,
            method.gan_cfg.model.noise_dim,
            x.columns,
            seed,
            device,
            condition,
        )

    x_aug, y_aug = add_synthetic(x, y, synthetic)

    return x_aug, y_aug, synthetic


def _evaluate_stage(
    methods: list[PreparedMethod],
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    scores_fit: pd.DataFrame | None,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    eval_name: str,
    seed: int,
    device: torch.device,
    with_quality: bool,
) -> StageResult:
    metric_rows: list[dict[str, float | str]] = []
    quality_rows: list[dict[str, float | str]] = []
    probabilities: dict[str, NDArray[np.float64]] = {}
    real_fraud = x_fit[y_fit == 1]

    for method in methods:
        x_aug, y_aug, synthetic = _training_set(method, x_fit, y_fit, scores_fit, seed, device)
        clf = fit_classifier(x_aug, y_aug, seed, method.cfg.classifier_params)
        _, fit_proba = predict_scores(clf, x_aug)
        _, eval_proba = predict_scores(clf, x_eval)

        label = method.cfg.label
        fit_metrics = ranking_metrics(y_aug, fit_proba)
        eval_metrics = ranking_metrics(y_eval, eval_proba)
        metric_rows.append({"method": label, "dataset": "train", **fit_metrics})
        metric_rows.append({"method": label, "dataset": eval_name, **eval_metrics})
        probabilities[label] = eval_proba

        if with_quality and synthetic is not None:
            quality_rows.append({"method": label, **synthetic_quality(real_fraud, synthetic)})

    return StageResult(pd.DataFrame(metric_rows), probabilities, pd.DataFrame(quality_rows))


def _false_alert_table(
    methods: list[PreparedMethod], stage: StageResult, y_eval: NDArray[np.int64]
) -> pd.DataFrame:
    baseline = next(m.cfg.label for m in methods if m.cfg.method == "baseline")
    counts = {
        label: false_alerts(RECALL_TARGETS, y_eval, proba)
        for label, proba in stage.probabilities.items()
    }
    rows = [
        {
            "recall_target": target,
            "method": label,
            "false_alerts": alerts[i],
            "saved_vs_baseline": counts[baseline][i] - alerts[i],
        }
        for label, alerts in counts.items()
        for i, target in enumerate(RECALL_TARGETS)
    ]

    return pd.DataFrame(rows)


def run_final_evaluation(
    configs: list[FinalMethodConfig], data: FinalData, seed: int, device: torch.device
) -> FinalResult:
    """
    Run both stages for every method. configs must include a baseline.
    """

    if not any(cfg.method == "baseline" for cfg in configs):
        raise ValueError("the comparison needs a baseline config to measure false alerts against")

    methods = [prepare_method(cfg, seed, device) for cfg in configs]
    target = data.target

    x_train, y_train = data.train.drop(columns=target), data.train[target]
    x_val, y_val = data.val.drop(columns=target), data.val[target]
    x_test, y_test = data.test.drop(columns=target), data.test[target]
    scores = data.scores

    val_stage = _evaluate_stage(
        methods,
        x_train,
        y_train,
        None if scores is None else scores["train"],
        x_val,
        y_val,
        "val",
        seed,
        device,
        with_quality=True,
    )

    x_full = pd.concat([x_train, x_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    scores_full = (
        None if scores is None else pd.concat([scores["train"], scores["val"]], ignore_index=True)
    )
    test_stage = _evaluate_stage(
        methods,
        x_full,
        y_full,
        scores_full,
        x_test,
        y_test,
        "test",
        seed,
        device,
        with_quality=False,
    )

    y_val_array = y_val.to_numpy(dtype=np.int64)
    y_test_array = y_test.to_numpy(dtype=np.int64)
    selected = pd.DataFrame(
        [
            {"method": m.cfg.label, "ratio": m.ratio, "selection": m.selection}
            for m in methods
            if m.ratio is not None
        ]
    )

    return FinalResult(
        selected_ratios=selected,
        val=val_stage,
        test=test_stage,
        y_val=y_val_array,
        y_test=y_test_array,
        false_alerts_val=_false_alert_table(methods, val_stage, y_val_array),
        false_alerts_test=_false_alert_table(methods, test_stage, y_test_array),
    )
