from pathlib import Path

from fraud_detection.config import load_dataset_config


def test_shipped_ieee_config_loads() -> None:
    cfg = load_dataset_config("ieee_cis")

    assert cfg.name == "ieee_cis"
    assert cfg.target == "isFraud"
    assert cfg.time_col == "TransactionDT"
    assert (cfg.split.val_size, cfg.split.test_size) == (0.2, 0.1)


def test_data_root_is_overridable(tmp_path: Path) -> None:
    cfg = load_dataset_config("ieee_cis", data_root=tmp_path)

    assert cfg.raw_dir == tmp_path / "ieee_cis" / "raw"
    assert cfg.splits_dir == tmp_path / "ieee_cis" / "splits"


def test_preprocessing_section_loads() -> None:
    pre = load_dataset_config("ieee_cis").preprocessing

    assert pre.max_col_missing == 0.55 and pre.max_row_missing == 0.48
    assert "P_emaildomain" in pre.qualitative_cols
