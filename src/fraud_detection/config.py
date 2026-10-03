from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.2
    test_size: float = 0.1


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: str
    raw_files: dict[str, str]
    target: str
    time_col: str
    id_cols: list[str]
    split: SplitConfig
    data_root: Path = DATA_DIR

    @property
    def raw_dir(self) -> Path:
        return self.data_root / self.name / "raw"

    @property
    def splits_dir(self) -> Path:
        return self.data_root / self.name / "splits"


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
        data_root=data_root or DATA_DIR,
    )
