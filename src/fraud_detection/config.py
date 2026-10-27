from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
FINAL_METHODS = ("baseline", "smote", "gan", "cgan")


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.2
    test_size: float = 0.1


@dataclass(frozen=True)
class PreprocessingConfig:
    max_col_missing: float = 0.50
    max_row_missing: float = 0.50
    qualitative_cols: list[str] = field(default_factory=list[str])
    already_encoded_cols: list[str] = field(default_factory=list[str])
    ohe_max_unique: int = 10
    skew_threshold: float = 0.5
    time_features: str = "seconds_offset"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: str
    raw_files: dict[str, str]
    target: str
    time_col: str
    id_cols: list[str]
    split: SplitConfig
    preprocessing: PreprocessingConfig
    data_root: Path = DATA_DIR

    @property
    def raw_dir(self) -> Path:
        return self.data_root / self.name / "raw"

    @property
    def splits_dir(self) -> Path:
        return self.data_root / self.name / "splits"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / self.name / "processed"


def load_dataset_config(name_or_path: str | Path, data_root: Path | None = None) -> DatasetConfig:
    """
    Load by dataset name or by explicit YAML path.
    """

    path = Path(name_or_path)

    if path.suffix not in {".yaml", ".yml"}:
        path = CONFIG_DIR / "datasets" / f"{name_or_path}.yaml"

    raw = yaml.safe_load(path.read_text())

    return DatasetConfig(
        name=raw["name"],
        loader=raw["loader"],
        raw_files=raw["raw_files"],
        target=raw["target"],
        time_col=raw["time_col"],
        id_cols=list(raw.get("id_cols", [])),
        split=SplitConfig(**raw.get("split", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        data_root=data_root or DATA_DIR,
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    One classifier, with and without SMOTE at several oversampling ratios.
    """

    name: str
    dataset: str
    seeds: list[int]
    classifier_params: dict[str, Any]
    smote_ratios: list[float]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text())

    return BenchmarkConfig(
        name=raw["name"],
        dataset=raw["dataset"],
        seeds=[int(seed) for seed in raw["seeds"]],
        classifier_params=dict(raw.get("classifier", {}).get("params") or {}),
        smote_ratios=[float(ratio) for ratio in raw["smote"]["ratios"]],
    )


@dataclass(frozen=True)
class GanModelConfig:
    noise_dim: int
    generator_hidden: list[int]
    critic_hidden: list[int]
    critic_dropout: float = 0.0
    critic_spectral_norm: bool = False


@dataclass(frozen=True)
class GanTrainingConfig:
    """
    objective: is bce (standard GAN with label smoothing) or wgan
                (critic + optional gradient penalty).
    n_critic: number of critics.
    lambda_gp: apply to wgan only.
    label_smoothing: (targets for real and fake) to bce only.
    real_noise_std: is added to real samples shown to the critic (0 disables it).
    """

    objective: str
    epochs: int
    g_lr: float
    d_lr: float
    batch_size: int = 256
    betas: tuple[float, float] = (0.5, 0.999)
    n_critic: int = 1
    lambda_gp: float = 0.0
    real_noise_std: float = 0.0
    label_smoothing: tuple[float, float] = (0.9, 0.1)
    lr_decay_epoch: int | None = None
    lr_decay_factor: float = 0.5


@dataclass(frozen=True)
class ConditionConfig:
    """
    Conditioning a GAN on an autoencoder anomaly score (turns the GAN into a cGAN).

    scores_from names the autoencoder experiment whose saved scores are used (for the same
    seed).
    n_bins equal-width bins stratify the scores drawn at generation time.
    """

    scores_from: str
    n_bins: int = 3


@dataclass(frozen=True)
class GanExperimentConfig:
    name: str
    dataset: str
    seeds: list[int]
    model: GanModelConfig
    training: GanTrainingConfig
    ratios: list[float]
    classifier_params: dict[str, Any]
    condition: ConditionConfig | None = None


def load_gan_config(path: str | Path) -> GanExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    training = dict(raw["training"])

    for key in ("betas", "label_smoothing"):
        if key in training:
            training[key] = (float(training[key][0]), float(training[key][1]))

    return GanExperimentConfig(
        name=raw["name"],
        dataset=raw["dataset"],
        seeds=[int(seed) for seed in raw["seeds"]],
        model=GanModelConfig(**raw["model"]),
        training=GanTrainingConfig(**training),
        ratios=[float(ratio) for ratio in raw["evaluation"]["ratios"]],
        classifier_params=dict(raw["evaluation"].get("classifier", {}).get("params") or {}),
        condition=ConditionConfig(**raw["condition"]) if raw.get("condition") else None,
    )


@dataclass(frozen=True)
class AutoencoderModelConfig:
    hidden: int
    latent: int
    dropout: float = 0.0


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    epochs: int
    lr: float
    batch_size: int = 256


@dataclass(frozen=True)
class AutoencoderExperimentConfig:
    name: str
    dataset: str
    seeds: list[int]
    model: AutoencoderModelConfig
    training: AutoencoderTrainingConfig


def load_autoencoder_config(path: str | Path) -> AutoencoderExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    return AutoencoderExperimentConfig(
        name=raw["name"],
        dataset=raw["dataset"],
        seeds=[int(seed) for seed in raw["seeds"]],
        model=AutoencoderModelConfig(**raw["model"]),
        training=AutoencoderTrainingConfig(**raw["training"]),
    )


@dataclass(frozen=True)
class FinalMethodConfig:
    """
    One method of the final comparison.
    """

    name: str
    label: str
    dataset: str
    method: str
    seeds: list[int]
    classifier_params: dict[str, Any]
    source: str | None = None
    ratio: float | None = None


def load_final_config(path: str | Path) -> FinalMethodConfig:
    raw = yaml.safe_load(Path(path).read_text())
    method = raw["method"]

    if method not in FINAL_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {FINAL_METHODS}")

    ratio = None if raw.get("ratio", "auto") == "auto" else float(raw["ratio"])
    source = raw.get("source")

    if method in ("gan", "cgan") and source is None:
        raise ValueError(
            f"{raw['name']}: a {method} needs `source`, the experiment with its generator"
        )

    if method == "smote" and ratio is None and source is None:
        raise ValueError(f"{raw['name']}: set `ratio`, or `source` to choose it from its results")

    return FinalMethodConfig(
        name=raw["name"],
        label=raw.get("label", raw["name"]),
        dataset=raw["dataset"],
        method=method,
        seeds=[int(seed) for seed in raw["seeds"]],
        classifier_params=dict(raw.get("classifier", {}).get("params") or {}),
        source=source,
        ratio=ratio,
    )


def find_experiment_config(name: str) -> Path:
    """
    The development config of the experiment called name (e.g. gan_08).
    """

    matches = sorted((CONFIG_DIR / "experiments" / "development").glob(f"*/{name}.yaml"))

    if len(matches) != 1:
        raise FileNotFoundError(f"expected one development config named {name}, found {matches}")

    return matches[0]
