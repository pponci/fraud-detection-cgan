"""
Usage: uv run python scripts/prepare_data.py --dataset ieee_cis
"""

import argparse

from fraud_detection.config import load_dataset_config
from fraud_detection.data.datasets import load_raw
from fraud_detection.data.splitting import temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ieee_cis", help="name in configs/datasets/")
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset)
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg.name}] loading raw data from {cfg.raw_dir}")
    raw = load_raw(cfg)
    print(f"  merged {raw.shape}")

    train, val, test = temporal_split(raw, cfg.split.val_size, cfg.split.test_size)

    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_parquet(cfg.splits_dir / f"{name}.parquet")
        print(f"  {name:<5} {df.shape}  fraud rate {df[cfg.target].mean():.4%}")


if __name__ == "__main__":
    main()
