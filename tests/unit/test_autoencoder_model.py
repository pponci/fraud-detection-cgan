import torch

from fraud_detection.models.autoencoder import AutoEncoder


def test_output_has_the_input_shape() -> None:
    model = AutoEncoder(in_dim=9, hidden=6, latent=3)

    assert model(torch.randn(5, 9)).shape == (5, 9)


def test_latent_code_has_the_requested_size() -> None:
    model = AutoEncoder(in_dim=9, hidden=6, latent=3)

    assert model.encoder(torch.randn(5, 9)).shape == (5, 3)


def test_layer_layout_without_dropout_matches_saved_checkpoints() -> None:
    keys = set(AutoEncoder(9, 6, 3).state_dict())

    assert keys == {
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
    }


def test_dropout_shifts_the_second_encoder_layer_to_index_three() -> None:
    keys = set(AutoEncoder(9, 6, 3, dropout=0.2).state_dict())

    assert {"encoder.0.weight", "encoder.3.weight", "decoder.0.weight", "decoder.2.weight"} <= keys
    assert "encoder.2.weight" not in keys
