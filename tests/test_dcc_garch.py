import numpy as np
import pandas as pd
import pytest

from src.dcc_garch import dcc_filter, fit_dcc, upper_triangle_features


def test_dcc_filter_obeys_correlation_matrix_invariants():
    rng = np.random.default_rng(10)
    residuals = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    matrices = dcc_filter(residuals, 0.04, 0.90)
    assert matrices.shape == (300, 3, 3)
    assert np.allclose(matrices, matrices.transpose(0, 2, 1))
    assert np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 1.0)
    assert min(np.linalg.eigvalsh(matrix).min() for matrix in matrices) > 0


@pytest.mark.parametrize(
    "a,b",
    [
        (-0.1, 0.5),
        (0.2, -0.1),
        (0.5, 0.5),
        (np.nan, 0.5),
        (0.2, np.inf),
    ],
)
def test_dcc_rejects_invalid_parameters(a, b):
    with pytest.raises(ValueError, match="a >= 0"):
        dcc_filter(np.ones((100, 2)), a, b)


def test_dcc_estimation_respects_parameter_constraints():
    rng = np.random.default_rng(12)
    common = rng.normal(size=250)
    values = np.column_stack(
        [common + rng.normal(scale=0.5, size=250), common + rng.normal(scale=0.6, size=250)]
    )
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    result = fit_dcc(
        pd.DataFrame(values, index=dates, columns=["bank1", "bank2"]), maxiter=100
    )
    assert result.converged
    assert result.a >= 0 and result.b >= 0 and result.a + result.b < 1
    assert result.correlations.shape == (250, 2, 2)
    assert result.observation_dates.equals(dates)


def test_upper_triangle_does_not_duplicate_pairs():
    matrices = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    features = upper_triangle_features(matrices)
    assert features.shape == (4, 3)


def test_dcc_uses_supplied_standardized_residuals_without_restandardizing():
    values = np.array([[2.0, 1.0], [-1.0, 3.0], [0.5, -2.0], [1.5, 0.25]])
    q_bar = values.T @ values / len(values)
    expected = q_bar / np.sqrt(np.outer(np.diag(q_bar), np.diag(q_bar)))
    actual = dcc_filter(values, 0.0, 0.0)[0]
    assert np.allclose(actual, expected)


def test_dcc_fit_rejects_duplicate_dates_and_constant_series():
    duplicated = pd.DataFrame(
        np.random.default_rng(1).normal(size=(40, 2)),
        index=[0] * 40,
        columns=["a", "b"],
    )
    with pytest.raises(ValueError, match="indices must be unique"):
        fit_dcc(duplicated)
    constant = pd.DataFrame({"a": np.ones(40), "b": np.arange(40, dtype=float)})
    with pytest.raises(ValueError, match="positive variance"):
        fit_dcc(constant)
