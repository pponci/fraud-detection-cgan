import csv
from pathlib import Path
from typing import Any, cast

import yaml

PLACEHOLDER = "not run"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def describe(config_path: Path) -> str:
    """
    The text after the colon in the config's first comment line ("# GAN iteration 3: ...").
    """

    first = config_path.read_text().splitlines()[0].lstrip("# ").strip()

    return first.split(": ", 1)[1] if ": " in first else first


def _load_mapping(path: Path) -> dict[str, Any]:
    """
    A YAML file as a dict; a file with only comments (or a non-mapping) gives {}.
    """

    loaded = yaml.safe_load(path.read_text())

    return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else {}


def _sources_in_use(final_dir: Path) -> set[str]:
    used: set[str] = set()

    for path in sorted(final_dir.glob("*.yaml")):
        source = _load_mapping(path).get("source")

        if source:
            used.add(source)

    return used


def _autoencoders_in_use(development: Path, used: set[str]) -> set[str]:
    scores: set[str] = set()

    for name in used:
        for path in development.glob(f"*/{name}.yaml"):
            condition = _load_mapping(path).get("condition")

            if condition:
                scores.add(condition["scores_from"])

    return scores


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join(lines)


def _mark(name: str, in_use: set[str]) -> str:
    return f"**{name}** (used)" if name in in_use else name


def _results(results_dir: Path, name: str, seed: int) -> list[dict[str, str]] | None:
    path = results_dir / name / f"seed_{seed}" / "metrics.csv"

    return read_rows(path) if path.exists() else None


def autoencoder_table(development: Path, results_dir: Path, seed: int, in_use: set[str]) -> str:
    rows: list[list[str]] = []

    for config in sorted((development / "ae").glob("*.yaml")):
        name = config.stem
        table = _results(results_dir, name, seed)

        if table is None:
            values = [PLACEHOLDER, PLACEHOLDER]

        else:
            sep = {r["dataset"]: float(r["sep_rate"]) for r in table}
            values = [f"{sep['train']:.3f}", f"{sep['val']:.3f}"]

        rows.append([_mark(name, in_use), describe(config), *values])

    return _table(["Experiment", "What changed", "Train separation", "Validation separation"], rows)


def generator_table(
    development: Path, results_dir: Path, seed: int, kind: str, in_use: set[str]
) -> str:
    rows: list[list[str]] = []

    for config in sorted((development / kind).glob("*.yaml")):
        name = config.stem
        table = _results(results_dir, name, seed)

        if table is None:
            values = [PLACEHOLDER, PLACEHOLDER]

        else:
            val = [r for r in table if r["dataset"] == "val"]
            best = max(val, key=lambda r: float(r["pr_auc"]))
            values = [f"{float(best['ratio']):g}", f"{float(best['pr_auc']):.4f}"]

        rows.append([_mark(name, in_use), describe(config), *values])

    return _table(["Experiment", "What changed", "Best ratio", "Best validation PR-AUC"], rows)


def smote_table(results_dir: Path, seed: int, in_use: set[str]) -> str:
    table = _results(results_dir, "smote_sweep", seed)

    if table is None:
        return f"SMOTE sweep: {PLACEHOLDER}."

    val = [r for r in table if r["dataset"] == "val"]
    rows = [
        [("baseline (no oversampling)" if r["method"] == "base" else f"{float(r['ratio']):g}"),
         f"{float(r['pr_auc']):.4f}"]
        for r in val
    ]  # fmt: skip

    return _table(["SMOTE ratio", "Validation PR-AUC"], rows)


HEADER = """# Experiments

Every experiment that was run during development, with the result it produced. This file is
**generated** from the committed results by `scripts/build_experiment_tables.py`; do not edit
it by hand. Each experiment is a config in `configs/experiments/development/` and can be re-run
with the script named in its section.

Experiments marked **(used)** are the ones the final comparison is built on
(`configs/experiments/final/`). All numbers are for seed {seed} on `{dataset}`.
"""

SECTIONS = (
    ("Autoencoders", "`scripts/train_autoencoder.py`", "autoencoder",
     "Trained on legitimate transactions only. *Separation* is the mean reconstruction error of "
     "fraud divided by that of legitimate transactions (higher is better), measured on raw "
     "errors."),
    ("GANs", "`scripts/train_gan.py`", "gan",
     "Each row shows the oversampling ratio with the best validation PR-AUC."),
    ("Conditional GANs", "`scripts/train_gan.py`", "cgan",
     "The GAN of iteration 8, conditioned on the anomaly score of the autoencoder marked "
     "(used) above."),
)  # fmt: skip


def render_experiments(dataset: str, seed: int, results_root: Path, configs_root: Path) -> str:
    development = configs_root / "development"
    results_dir = results_root / dataset
    used = _sources_in_use(configs_root / "final" / dataset)
    used_autoencoders = _autoencoders_in_use(development, used)

    parts = [HEADER.format(seed=seed, dataset=dataset)]

    for title, script, kind, note in SECTIONS:
        parts.append(f"## {title}\n\nRun with {script}. {note}\n")

        if kind == "autoencoder":
            parts.append(autoencoder_table(development, results_dir, seed, used_autoencoders))

        else:
            parts.append(generator_table(development, results_dir, seed, kind, used))

        parts.append("")

    parts.append(
        "## SMOTE\n\nRun with `scripts/run_benchmark.py`. Validation PR-AUC of the baseline and "
        "of SMOTE at each ratio (fraud / legitimate after oversampling).\n"
    )

    parts.append(smote_table(results_dir, seed, used))

    return "\n".join(parts) + "\n"
