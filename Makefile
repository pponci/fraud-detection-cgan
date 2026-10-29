.PHONY: setup check lint format typecheck test tables reproduce

setup:  ## install dependencies and the notebook output-stripping filter
	uv sync
	uv run nbstripout --install --attributes .gitattributes

check: lint typecheck test  ## everything CI runs

lint:
	uv run ruff format --check .
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

test:
	uv run pytest tests/unit --cov=fraud_detection --cov-report=term-missing

tables:  ## regenerate docs/experiments.md from the committed results
	uv run python scripts/build_experiment_tables.py

reproduce:  ## the final pipeline, from raw data to the final tables (slow)
	uv run python scripts/prepare_data.py --dataset ieee_cis
	uv run python scripts/run_benchmark.py --config configs/experiments/development/benchmark/smote_sweep.yaml
	uv run python scripts/train_autoencoder.py --config configs/experiments/development/ae/ae_08.yaml
	uv run python scripts/train_gan.py --config configs/experiments/development/gan/gan_08.yaml
	uv run python scripts/train_gan.py --config configs/experiments/development/cgan/cgan_04.yaml
	uv run python scripts/evaluate.py
	uv run python scripts/build_experiment_tables.py