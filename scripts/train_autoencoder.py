"""
Usage:
    uv run python scripts/train_autoencoder.py \
        --config configs/experiments/development/ae/ae_08.yaml

To evaluate an autoencoder you already have, instead of training one:
    uv run python scripts/train_autoencoder.py --config <config> --checkpoint path/to/model.pth
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from fraud_detection.config import (
    ARTIFACTS_DIR,
    RESULTS_DIR,
    load_autoencoder_config,
    load_dataset_config,
)
from fraud_detection.training.autoencoder_experiment import run_autoencoder_experiment
from fraud_detection.utils.seed import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="evaluate this model, do not train")
    parser.add_argument("--force", action="store_true", help="recompute existing results")
    args = parser.parse_args()

    cfg = load_autoencoder_config(args.config)
    dataset = load_dataset_config(cfg.dataset)
    train = pd.read_parquet(dataset.processed_dir / "train.parquet")
    val = pd.read_parquet(dataset.processed_dir / "val.parquet")
    test = pd.read_parquet(dataset.processed_dir / "test.parquet")
    device = get_device()
    print(f"device: {device}")

    for seed in cfg.seeds:
        artifacts_dir = ARTIFACTS_DIR / dataset.name / cfg.name / f"seed_{seed}"
        results_dir = RESULTS_DIR / dataset.name / cfg.name / f"seed_{seed}"

        if args.checkpoint:
            checkpoint = args.checkpoint
            artifacts_out = results_out = artifacts_dir / "checkpoint_eval"

        else:
            checkpoint, artifacts_out, results_out = (
                artifacts_dir / "ae.pt",
                artifacts_dir,
                results_dir,
            )

        if (results_out / "metrics.csv").exists() and not args.force:
            print(f"[seed {seed}] results exist in {results_out}, skipping (use --force)")
            continue

        if args.force and not args.checkpoint:
            checkpoint.unlink(missing_ok=True)

        print(f"[seed {seed}] {cfg.name} on {dataset.name}")
        result = run_autoencoder_experiment(
            train, val, test, dataset.target, cfg, seed, device, checkpoint
        )

        results_out.mkdir(parents=True, exist_ok=True)
        artifacts_out.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(results_out / "metrics.csv", index=False)
        result.history.to_csv(results_out / "history.csv")
        shutil.copy(args.config, results_out / "config.yaml")
        for split, scores in result.scores.items():
            scores.to_parquet(artifacts_out / f"scores_{split}.parquet")

        print(result.summary.round(4).to_string(index=False))
        print(f"[seed {seed}] results in {results_out}, scores in {artifacts_out}")


if __name__ == "__main__":
    main()
