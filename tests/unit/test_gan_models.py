import torch

from fraud_detection.models.gan import Critic, Generator


def test_generator_output_shape() -> None:
    generator = Generator(data_dim=7, noise_dim=4, hidden=[8, 6])

    assert generator(torch.randn(5, 4)).shape == (5, 7)


def test_critic_output_is_one_score_per_sample() -> None:
    critic = Critic(data_dim=7, hidden=[8, 6])

    assert critic(torch.randn(5, 7)).shape == (5, 1)


def test_generator_layer_layout_matches_saved_checkpoints() -> None:
    keys = set(Generator(7, 4, [8, 6]).state_dict())

    assert {"fc.0.weight", "fc.1.weight", "fc.3.weight", "fc.4.weight", "fc.6.weight"} <= keys


def test_critic_layer_layout_with_dropout_matches_saved_checkpoints() -> None:
    keys = set(Critic(7, [8, 6], dropout=0.3).state_dict())

    assert keys == {
        "fc.0.weight",
        "fc.0.bias",
        "fc.3.weight",
        "fc.3.bias",
        "fc.6.weight",
        "fc.6.bias",
    }


def test_critic_layer_layout_with_spectral_norm_matches_saved_checkpoints() -> None:
    keys = set(Critic(7, [8, 6], use_spectral_norm=True).state_dict())

    assert {"fc.0.weight_orig", "fc.0.weight_u", "fc.2.weight_orig", "fc.4.weight_orig"} <= keys
