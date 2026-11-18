import numpy as np
import pytest

from src.ml import (
    CorrelationAutoencoder,
    reconstruction_errors,
    upper_triangle_features,
)


def test_model_accepts_and_reconstructs_ten_upper_triangle_features():
    model = CorrelationAutoencoder(input_dim=10, latent_dim=3)
    features = np.random.default_rng(7).normal(size=(32, 10)).astype(np.float32)
    reconstructed = model.reconstruct(features)

    assert features.shape == (32, 10)
    assert reconstructed.shape == (32, 10)
    assert np.isfinite(reconstructed).all()


def test_reconstruction_errors_have_one_score_per_observation():
    model = CorrelationAutoencoder(input_dim=10, latent_dim=3)
    features = np.random.default_rng(11).normal(size=(17, 10)).astype(np.float32)
    errors = reconstruction_errors(model, features)

    assert errors.shape == (17,)
    assert np.isfinite(errors).all()
    assert (errors >= 0).all()


def test_upper_triangle_extraction_has_ten_features_for_five_assets():
    rng = np.random.default_rng(13)
    matrices = np.empty((6, 5, 5), dtype=float)
    for i in range(6):
        base = rng.normal(size=(5, 5))
        matrices[i] = np.eye(5) * 0.8 + base @ base.T * 0.01

    features = upper_triangle_features(matrices)
    assert features.shape == (6, 10)


@pytest.mark.parametrize("bad_features", [np.ones(10), np.ones((2, 9)), np.ones((2, 11))])
def test_model_rejects_wrong_feature_dimensions(bad_features):
    model = CorrelationAutoencoder(input_dim=10)
    with pytest.raises(ValueError, match="features|2-D|expected"):
        model.reconstruct(bad_features)
