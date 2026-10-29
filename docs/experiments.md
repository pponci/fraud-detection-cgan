# Experiments

Every experiment that was run during development, with the result it produced. This file is
**generated** from the committed results by `scripts/build_experiment_tables.py`; do not edit
it by hand. Each experiment is a config in `configs/experiments/development/` and can be re-run
with the script named in its section.

Experiments marked **(used)** are the ones the final comparison is built on
(`configs/experiments/final/`). All numbers are for seed 11 on `ieee_cis`.

## Autoencoders

Run with `scripts/train_autoencoder.py`. Trained on legitimate transactions only. *Separation* is the mean reconstruction error of fraud divided by that of legitimate transactions (higher is better), measured on raw errors.

| Experiment | What changed | Train separation | Validation separation |
|---|---|---|---|
| ae_01 | ae_01 | 2.740 | 1.383 |
| ae_02 | ae_02 | 3.166 | 0.951 |
| ae_03 | ae_03 | 2.940 | 1.644 |
| ae_04 | ae_04 | 3.331 | 1.279 |
| ae_05 | ae_05 | 3.280 | 1.478 |
| ae_06 | ae_06 | 3.005 | 1.477 |
| ae_07 | ae_07 | 2.896 | 1.599 |
| **ae_08** (used) | ae_08 | 3.767 | 2.541 |

## GANs

Run with `scripts/train_gan.py`. Each row shows the oversampling ratio with the best validation PR-AUC.

| Experiment | What changed | Best ratio | Best validation PR-AUC |
|---|---|---|---|
| gan_01 | gan_01 | 0.15 | 0.5127 |
| gan_02 | gan_02 | 0.05 | 0.5236 |
| gan_03 | gan_03 | 0.1 | 0.5151 |
| gan_04 | gan_04 | 0.05 | 0.5174 |
| gan_05 | gan_05 | 0.05 | 0.5190 |
| gan_06 | gan_06 | 0.075 | 0.5157 |
| gan_07 | gan_07 | 0.05 | 0.5194 |
| **gan_08** (used) | gan_08 | 0.1 | 0.5187 |

## Conditional GANs

Run with `scripts/train_gan.py`. The GAN of iteration 8, conditioned on the anomaly score of the autoencoder marked (used) above.

| Experiment | What changed | Best ratio | Best validation PR-AUC |
|---|---|---|---|
| cgan_01 | cgan_01 | 0.1 | 0.4734 |
| cgan_02 | cgan_02 | 0.05 | 0.5249 |
| cgan_03 | cgan_03 | 0.075 | 0.5068 |
| **cgan_04** (used) | cgan_04 | 0.1 | 0.5185 |

## SMOTE

Run with `scripts/run_benchmark.py`. Validation PR-AUC of the baseline and of SMOTE at each ratio (fraud / legitimate after oversampling).

| SMOTE ratio | Validation PR-AUC |
|---|---|
| baseline (no oversampling) | 0.5076 |
| 0.05 | 0.5178 |
| 0.075 | 0.5143 |
| 0.1 | 0.5034 |
| 0.15 | 0.4920 |
| 0.2 | 0.4877 |
| 0.25 | 0.4849 |
| 0.5 | 0.4712 |
