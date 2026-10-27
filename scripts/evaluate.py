"""
Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --dataset ieee_cis --force
"""

import argparse
import shutil

import pandas as pd

from fraud_detection.config import (
    ARTIFACTS_DIR,
    CONFIG_DIR,
    FINAL_METHODS,
    RESULTS_DIR,
    FinalMethodConfig,
    find_experiment_config,
    load_dataset_config,
    load_final_config,
    load_gan_config,
)
from fraud_detection.evaluation.report import write_report
from fraud_detection.training.final_evaluation import FinalData, run_final_evaluation
from fraud_detection.utils.seed import get_device


def load_scores(
    configs: list[FinalMethodConfig], dataset: str, seed: int
) -> dict[str, pd.DataFrame] | None:
    """
    Anomaly scores of every split, if the comparison includes a cGAN.
    """

    sources = {
        load_gan_config(find_experiment_config(cfg.source)).condition
        for cfg in configs
        if cfg.method == "cgan" and cfg.source is not None
    }

    sources.discard(None)

    if not sources:
        return None

    (condition,) = sources

    assert condition is not None

    folder = ARTIFACTS_DIR / dataset / condition.scores_from / f"seed_{seed}"
    missing = [s for s in ("train", "val", "test") if not (folder / f"scores_{s}.parquet").exists()]

    if missing:
        raise SystemExit(
            f"missing anomaly scores {missing} in {folder}: run scripts/train_autoencoder.py "
            f"for {condition.scores_from} (seed {seed}) first"
        )

    return {s: pd.read_parquet(folder / f"scores_{s}.parquet") for s in ("train", "val", "test")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ieee_cis")
    parser.add_argument("--force", action="store_true", help="recompute existing results")
    args = parser.parse_args()

    paths = sorted((CONFIG_DIR / "experiments" / "final" / args.dataset).glob("*.yaml"))
    configs = sorted(
        (load_final_config(path) for path in paths), key=lambda c: FINAL_METHODS.index(c.method)
    )

    if not configs:
        raise SystemExit(f"no final configs found for dataset {args.dataset}")
    seeds = configs[0].seeds

    if any(cfg.seeds != seeds for cfg in configs):
        raise SystemExit("all final configs must list the same seeds")

    dataset = load_dataset_config(args.dataset)
    splits = {
        s: pd.read_parquet(dataset.processed_dir / f"{s}.parquet") for s in ("train", "val", "test")
    }
    device = get_device()
    print(f"device: {device}")

    for seed in seeds:
        results_dir = RESULTS_DIR / dataset.name / "final" / f"seed_{seed}"

        if (results_dir / "test_metrics.csv").exists() and not args.force:
            print(f"[seed {seed}] results exist in {results_dir}, skipping (use --force)")
            continue

        print(f"[seed {seed}] final evaluation on {dataset.name}: {[c.label for c in configs]}")
        data = FinalData(
            splits["train"],
            splits["val"],
            splits["test"],
            dataset.target,
            load_scores(configs, dataset.name, seed),
        )
        result = run_final_evaluation(configs, data, seed, device)

        artifacts_dir = ARTIFACTS_DIR / dataset.name / "final" / f"seed_{seed}"
        write_report(result, results_dir, artifacts_dir)

        for path in paths:
            (results_dir / "configs").mkdir(exist_ok=True)
            shutil.copy(path, results_dir / "configs" / path.name)

        print(result.selected_ratios.to_string(index=False))

        for name, stage in (("validation", result.val), ("test", result.test)):
            rows = stage.metrics[stage.metrics["dataset"] == name.replace("validation", "val")]
            print(f"\n{name} set\n{rows.round(4).to_string(index=False)}")
        print(f"\n[seed {seed}] written to {results_dir}")


if __name__ == "__main__":
    main()
