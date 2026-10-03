# Data

Nothing in this folder is committed except this file.

## Layout

Per dataset, under `data/<dataset>/`:

- `raw/` : files exactly as downloaded
- `splits/` : temporal train/val/test tables (merged, not yet cleaned), parquet
- `processed/` : cleaned, encoded and scaled tables (parquet) plus `preprocessor.joblib`,
  the fitted transformer that reproduces the exact same processing on new data

## IEEE-CIS Fraud Detection (`ieee_cis`)

Source: Kaggle competition "IEEE-CIS Fraud Detection",
https://www.kaggle.com/competitions/ieee-fraud-detection/data
You must accept the competition rules to download it, and the data cannot be redistributed.

Only two files are used:

- `train_transaction.csv` (590,540 rows, 394 columns)
- `train_identity.csv` (144,233 rows, 41 columns)

The competition's test files have no labels and are not used. Place both files in
`data/ieee_cis/raw/`, then build everything with:

    uv run python scripts/prepare_data.py --dataset ieee_cis

Expected shapes after the build:

| Split | After merge and split | After processing |
|-------|-----------------------|------------------|
| train | (413378, 434)         | (413165, 237)    |
| val   | (118108, 434)         | (118107, 237)    |
| test  | (59054, 434)          | (59053, 237)     |

## Adding another dataset

Add `configs/datasets/<name>.yaml`, register a loader in
`src/fraud_detection/data/datasets.py`, and put the raw files in `data/<name>/raw/`.