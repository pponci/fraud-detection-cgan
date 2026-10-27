from pathlib import Path

import pandas as pd

from fraud_detection.evaluation.plots import PALETTE, plot_false_alerts_saved, plot_pr_curves
from fraud_detection.evaluation.ranking import RECALL_TARGETS
from fraud_detection.training.final_evaluation import FinalResult, StageResult


def _pr_aucs(stage: StageResult, eval_name: str) -> dict[str, float]:
    rows = stage.metrics[stage.metrics["dataset"] == eval_name]

    return dict(zip(rows["method"], rows["pr_auc"].astype(float), strict=True))


def _saved(false_alerts: pd.DataFrame, baseline: str) -> dict[str, list[int]]:
    methods = false_alerts["method"].to_numpy(dtype=str)
    values = false_alerts["saved_vs_baseline"].to_numpy(dtype=int)

    return {
        label: values[methods == label].tolist()
        for label in dict.fromkeys(methods.tolist())
        if label != baseline
    }


def write_report(result: FinalResult, results_dir: Path, artifacts_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures = results_dir / "figures"

    result.selected_ratios.to_csv(results_dir / "selected_ratios.csv", index=False)
    result.val.metrics.to_csv(results_dir / "val_metrics.csv", index=False)
    result.test.metrics.to_csv(results_dir / "test_metrics.csv", index=False)
    result.val.quality.to_csv(results_dir / "synthetic_metrics.csv", index=False)
    result.false_alerts_val.to_csv(results_dir / "false_alerts_val.csv", index=False)
    result.false_alerts_test.to_csv(results_dir / "false_alerts_test.csv", index=False)

    colors = dict(zip(result.val.probabilities, PALETTE, strict=False))  # one colour per method
    augmented = set(result.selected_ratios["method"])
    baseline = next(label for label in result.val.probabilities if label not in augmented)

    for name, stage, y, alerts in (
        ("val", result.val, result.y_val, result.false_alerts_val),
        ("test", result.test, result.y_test, result.false_alerts_test),
    ):
        plot_pr_curves(
            y,
            stage.probabilities,
            _pr_aucs(stage, name),
            colors,
            f"Precision-recall curves ({name} set)",
            figures / f"pr_curves_{name}.png",
        )
        plot_false_alerts_saved(
            RECALL_TARGETS,
            _saved(alerts, baseline),
            colors,
            f"False alerts saved against the baseline ({name} set)",
            figures / f"false_alerts_{name}.png",
        )
        predictions = pd.DataFrame({"y_true": y, **stage.probabilities})
        predictions.to_parquet(artifacts_dir / f"predictions_{name}.parquet")
