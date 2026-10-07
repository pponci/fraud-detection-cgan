"""
Usage:
    uv run python scripts/run_benchmark.py \
        --config configs/experiments/development/benchmark/smote_sweep.yaml
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from fraud_detection.config import RESULTS_DIR, load_benchmark_config, load_dataset_config
from fraud_detection.training.benchmark import run_benchmark
from fraud_detection.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="recompute existing seeds")
    args = parser.parse_args()

    bench = load_benchmark_config(args.config)
    dataset = load_dataset_config(bench.dataset)
    train = pd.read_parquet(dataset.processed_dir / "train.parquet")
    val = pd.read_parquet(dataset.processed_dir / "val.parquet")

    for seed in bench.seeds:
        out_dir = RESULTS_DIR / dataset.name / bench.name / f"seed_{seed}"
        if (out_dir / "metrics.csv").exists() and not args.force:
            print(f"[seed {seed}] results exist in {out_dir}, skipping (use --force)")
            continue

        print(f"[seed {seed}] running {bench.name} on {dataset.name}")
        set_seed(seed)
        metrics, synthetic = run_benchmark(
            train, val, dataset.target, bench.smote_ratios, seed, bench.classifier_params
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_dir / "metrics.csv", index=False)
        synthetic.to_csv(out_dir / "synthetic_metrics.csv", index=False)
        shutil.copy(args.config, out_dir / "config.yaml")
        print(metrics[metrics["dataset"] == "val"].round(4).to_string(index=False))
        print(f"[seed {seed}] written to {out_dir}")


if __name__ == "__main__":
    main()
