import numpy as np
import pandas as pd
import pytest

from src.var import historical_var, parametric_var


def test_var_estimators_return_positive_loss_values():
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(-0.001, 0.02, 1000))
    assert historical_var(returns, 0.05) > 0
    assert parametric_var(returns, 0.05) > 0


def test_var_validates_tail_probability_and_observations():
    with pytest.raises(ValueError, match="alpha"):
        historical_var(pd.Series(np.arange(50)), 0.8)
    with pytest.raises(ValueError, match="20"):
        historical_var(pd.Series([0.01, -0.01]), 0.05)


def test_parametric_var_rejects_zero_volatility():
    with pytest.raises(ValueError, match="standard deviation"):
        parametric_var(pd.Series(np.ones(100) * 0.01))


def test_parametric_var_accepts_genuinely_small_nonzero_volatility():
    returns = pd.Series(np.array([0.0, 1e-12, 2e-12, 3e-12]))
    result = parametric_var(returns, alpha=0.05)
    assert np.isfinite(result)
    assert result > 0


def test_parametric_var_accepts_explicit_positive_small_standard_deviation():
    returns = pd.Series([0.0, 0.01])
    assert np.isfinite(parametric_var(returns, std=1e-20))


def test_var_known_small_sample_quantile():
    returns = pd.Series([-0.10] * 5 + [0.0] * 15 + [0.10] * 30)
    # NumPy's default linear quantile interpolates between -0.10 and 0.0.
    assert np.isclose(historical_var(returns, alpha=0.10), 0.01)


def test_historical_var_is_positive_when_lower_tail_is_a_loss():
    returns = pd.Series([-0.08, -0.05, -0.03, -0.01] + [0.01] * 20)
    assert historical_var(returns, alpha=0.05) > 0


def test_historical_var_keeps_sign_if_selected_quantile_is_a_gain():
    returns = pd.Series([0.01, 0.02, 0.03, 0.04] * 10)
    assert historical_var(returns, alpha=0.05) < 0
