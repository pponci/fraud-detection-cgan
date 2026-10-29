"""
Usage: uv run python scripts/build_experiment_tables.py [--dataset ieee_cis] [--seed 11]
"""

import argparse

from fraud_detection.config import CONFIG_DIR, REPO_ROOT, RESULTS_DIR
from fraud_detection.evaluation.experiment_tables import render_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ieee_cis")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    text = render_experiments(args.dataset, args.seed, RESULTS_DIR, CONFIG_DIR / "experiments")
    out = REPO_ROOT / "docs" / "experiments.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
