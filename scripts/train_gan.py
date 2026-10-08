"""
Usage:
    uv run python scripts/train_gan.py --config configs/experiments/development/gan/gan_01.yaml

To evaluate a generator you already have, instead of training one:
    uv run python scripts/train_gan.py --config <config> --checkpoint path/to/generator.pth
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from fraud_detection.config import (
    ARTIFACTS_DIR,
    RESULTS_DIR,
    load_dataset_config,
    load_gan_config,
)
from fraud_detection.training.gan_experiment import run_gan_experiment
from fraud_detection.utils.seed import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="evaluate this generator, do not train")
    parser.add_argument("--force", action="store_true", help="recompute existing results")
    args = parser.parse_args()

    cfg = load_gan_config(args.config)
    dataset = load_dataset_config(cfg.dataset)
    train = pd.read_parquet(dataset.processed_dir / "train.parquet")
    val = pd.read_parquet(dataset.processed_dir / "val.parquet")
    device = get_device()

    print(f"device: {device}")

    for seed in cfg.seeds:
        artifacts_dir = ARTIFACTS_DIR / dataset.name / cfg.name / f"seed_{seed}"

        if args.checkpoint:
            checkpoint, out_dir = args.checkpoint, artifacts_dir / "checkpoint_eval"

        else:
            checkpoint, out_dir = artifacts_dir / "gan.pt", RESULTS_DIR / dataset.name / cfg.name
            out_dir = out_dir / f"seed_{seed}"

        if (out_dir / "metrics.csv").exists() and not args.force:
            print(f"[seed {seed}] results exist in {out_dir}, skipping (use --force)")
            continue

        if args.force and not args.checkpoint:
            checkpoint.unlink(missing_ok=True)

        print(f"[seed {seed}] {cfg.name} on {dataset.name}")
        result = run_gan_experiment(train, val, dataset.target, cfg, seed, device, checkpoint)

        out_dir.mkdir(parents=True, exist_ok=True)
        result.metrics.to_csv(out_dir / "metrics.csv", index=False)
        result.synthetic_quality.to_csv(out_dir / "synthetic_metrics.csv", index=False)
        result.history.to_csv(out_dir / "history.csv")
        shutil.copy(args.config, out_dir / "config.yaml")

        val_rows = result.metrics[result.metrics["dataset"] == "val"]
        print(val_rows.round(4).to_string(index=False))
        print(f"[seed {seed}] written to {out_dir}")


if __name__ == "__main__":
    main()
