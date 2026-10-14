from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fraud_detection.config import GanModelConfig, GanTrainingConfig
from fraud_detection.training.gan import gradient_penalty, train_gan

CPU = torch.device("cpu")
MODEL = GanModelConfig(
    noise_dim=8, generator_hidden=[16, 16], critic_hidden=[16], critic_dropout=0.1
)
BCE = GanTrainingConfig(objective="bce", epochs=2, g_lr=1e-3, d_lr=1e-3, batch_size=32)
WGAN = GanTrainingConfig(
    objective="wgan",
    epochs=2,
    g_lr=1e-3,
    d_lr=1e-3,
    batch_size=32,
    betas=(0.5, 0.9),
    n_critic=2,
    lambda_gp=1.0,
    real_noise_std=0.01,
)


@pytest.fixture
def fraud() -> pd.DataFrame:
    rng = np.random.default_rng(0)

    return pd.DataFrame(rng.normal(size=(100, 6)), columns=[f"f{i}" for i in range(6)])


def test_gradient_penalty_of_a_linear_critic() -> None:
    critic = nn.Linear(2, 1)

    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 4.0]]))

    real, fake = torch.randn(4, 2), torch.randn(4, 2)

    assert gradient_penalty(critic, real, fake, CPU).item() == pytest.approx((5 - 1) ** 2)


@pytest.mark.parametrize("training", [BCE, WGAN], ids=["bce", "wgan"])
def test_training_runs_and_records_finite_losses(
    fraud: pd.DataFrame, training: GanTrainingConfig
) -> None:
    trained = train_gan(fraud, MODEL, training, seed=1, device=CPU)

    assert len(trained.d_losses) == len(trained.g_losses) == training.epochs
    assert np.isfinite(trained.d_losses + trained.g_losses).all()
    assert trained.generator(torch.randn(3, MODEL.noise_dim)).shape == (3, fraud.shape[1])


def test_spectral_norm_critic_trains(fraud: pd.DataFrame) -> None:
    model = replace(MODEL, critic_dropout=0.0, critic_spectral_norm=True)
    trained = train_gan(fraud, model, replace(WGAN, lambda_gp=0.0, n_critic=1), 1, CPU)

    assert np.isfinite(trained.d_losses).all()


def test_same_seed_gives_identical_generators(fraud: pd.DataFrame) -> None:
    first = train_gan(fraud, MODEL, WGAN, seed=3, device=CPU)
    second = train_gan(fraud, MODEL, WGAN, seed=3, device=CPU)
    other = train_gan(fraud, MODEL, WGAN, seed=4, device=CPU)

    first_state, second_state = first.generator.state_dict(), second.generator.state_dict()

    assert all(torch.equal(first_state[k], second_state[k]) for k in first_state)
    assert first.d_losses == second.d_losses
    assert not torch.equal(first_state["fc.0.weight"], other.generator.state_dict()["fc.0.weight"])


def test_unknown_objective_is_rejected(fraud: pd.DataFrame) -> None:

    with pytest.raises(ValueError, match="objective"):
        train_gan(fraud, MODEL, replace(BCE, objective="lsgan"), seed=1, device=CPU)


def test_too_few_rows_for_one_batch_is_a_clear_error(fraud: pd.DataFrame) -> None:

    with pytest.raises(ValueError, match="cannot fill one batch"):
        train_gan(fraud.iloc[:10], MODEL, BCE, seed=1, device=CPU)


def make_condition(n: int) -> np.ndarray:
    return np.random.default_rng(1).normal(size=(n, 1)).astype(np.float32)


def test_conditional_training_runs_and_builds_conditional_networks(fraud: pd.DataFrame) -> None:
    trained = train_gan(fraud, MODEL, WGAN, 1, CPU, condition=make_condition(len(fraud)))

    assert trained.cond_dim == 1
    assert np.isfinite(trained.d_losses + trained.g_losses).all()

    noise, cond = torch.randn(3, MODEL.noise_dim), torch.randn(3, 1)

    assert trained.generator(noise, cond).shape == (3, fraud.shape[1])


def test_the_condition_pairing_changes_what_is_learned(fraud: pd.DataFrame) -> None:
    condition = make_condition(len(fraud))
    first = train_gan(fraud, MODEL, WGAN, 3, CPU, condition=condition)
    second = train_gan(fraud, MODEL, WGAN, 3, CPU, condition=condition[::-1].copy())
    same = torch.equal(
        first.generator.state_dict()["fc.0.weight"], second.generator.state_dict()["fc.0.weight"]
    )

    assert not same


def test_conditioning_is_only_available_for_the_wgan_objective(fraud: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="only implemented for the 'wgan'"):
        train_gan(fraud, MODEL, BCE, 1, CPU, condition=make_condition(len(fraud)))


def test_learning_rates_are_scaled_once_at_the_decay_epoch(fraud: pd.DataFrame) -> None:
    decayed = train_gan(fraud, MODEL, replace(WGAN, epochs=3, lr_decay_epoch=1), 1, CPU)

    assert decayed.g_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.5)
    assert decayed.d_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.5)

    plain = train_gan(fraud, MODEL, replace(WGAN, epochs=3), 1, CPU)

    assert plain.g_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
