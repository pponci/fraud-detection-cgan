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


def test_conditional_networks_take_the_condition_as_extra_input() -> None:
    generator = Generator(data_dim=7, noise_dim=4, hidden=[8], cond_dim=1)
    critic = Critic(data_dim=7, hidden=[8], cond_dim=1)
    noise, cond, x = torch.randn(5, 4), torch.randn(5, 1), torch.randn(5, 7)

    assert generator(noise, cond).shape == (5, 7)
    assert critic(x, cond).shape == (5, 1)
    assert generator.state_dict()["fc.0.weight"].shape == (8, 4 + 1)
    assert critic.state_dict()["fc.0.weight"].shape == (8, 7 + 1)


def test_the_condition_changes_the_output() -> None:
    generator = Generator(data_dim=7, noise_dim=4, hidden=[8], cond_dim=1).eval()
    noise = torch.randn(5, 4)
    low, high = torch.full((5, 1), -2.0), torch.full((5, 1), 2.0)

    assert not torch.allclose(generator(noise, low), generator(noise, high))


def test_conditioning_does_not_change_the_checkpoint_layout() -> None:
    assert set(Generator(7, 4, [8, 6], cond_dim=1).state_dict()) == set(
        Generator(7, 4, [8, 6]).state_dict()
    )
