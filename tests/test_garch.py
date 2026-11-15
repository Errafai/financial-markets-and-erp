import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.garch import (
    GARCHFitError,
    GARCHResult,
    align_standardized_residuals,
    fit_garch,
)
from src.synthetic_data import generate_synthetic_data


def test_garch_returns_valid_volatility_and_standardized_residuals():
    _, returns, _, _ = generate_synthetic_data()
    result = fit_garch(returns["ATW_SIM"])
    assert result.diagnostics["converged"] is True
    assert result.diagnostics["admissible"] is True
    assert (result.conditional_volatility > 0).all()
    assert np.isfinite(result.standardized_residuals).all()
    assert len(result.standardized_residuals) == len(returns)
    assert result.diagnostics["alpha_plus_beta"] < 1
    assert "standardized_residual_ljung_box_pvalue_lag10" in result.diagnostics
    assert "standardized_residual_jarque_bera_pvalue" in result.diagnostics


def test_garch_rejects_short_or_constant_input():
    with pytest.raises(ValueError, match="100"):
        fit_garch(pd.Series(np.zeros(50)))
    with pytest.raises(ValueError, match="non-zero variance"):
        fit_garch(pd.Series(np.ones(150)))


def test_garch_rejects_orders_outside_project_model():
    series = pd.Series(np.random.default_rng(7).normal(size=150))
    with pytest.raises(ValueError, match=re.escape("GARCH(1,1) only")):
        fit_garch(series, p=2, q=1)


def test_garch_panel_residuals_align_on_common_dates_without_filling():
    dates = pd.bdate_range("2024-01-01", periods=40)
    first_dates = dates.delete(5)
    second_dates = dates.delete(12)
    first = pd.Series(np.arange(len(first_dates), dtype=float), index=first_dates)
    second = pd.Series(np.arange(len(second_dates), dtype=float) + 1, index=second_dates)
    first.loc[dates[15]] = np.inf

    def result(series: pd.Series) -> GARCHResult:
        return GARCHResult(
            parameters=pd.Series(dtype=float),
            conditional_volatility=pd.Series(dtype=float),
            standardized_residuals=series,
            diagnostics={"converged": True, "admissible": True},
            fitted_model=object(),
        )

    aligned = align_standardized_residuals(
        {"bank_a": result(first), "bank_b": result(second)}
    )
    expected_dates = first_dates.intersection(second_dates).difference([dates[15]])
    expected_dates = expected_dates.sort_values()
    assert aligned.index.equals(expected_dates)
    assert len(aligned) == 37
    assert np.isfinite(aligned.to_numpy()).all()


def test_dcc_alignment_rejects_inadmissible_garch_result():
    dates = pd.bdate_range("2024-01-01", periods=40)
    residuals = pd.Series(np.arange(40, dtype=float), index=dates)
    valid = GARCHResult(
        parameters=pd.Series(dtype=float),
        conditional_volatility=pd.Series(dtype=float),
        standardized_residuals=residuals,
        diagnostics={"converged": True, "admissible": True},
        fitted_model=object(),
    )
    invalid = GARCHResult(
        parameters=pd.Series(dtype=float),
        conditional_volatility=pd.Series(dtype=float),
        standardized_residuals=residuals,
        diagnostics={"converged": True, "admissible": False},
        fitted_model=object(),
    )
    with pytest.raises(ValueError, match="not admissible"):
        align_standardized_residuals({"valid": valid, "invalid": invalid})


def test_garch_rejects_nonstationary_fit_and_preserves_estimates():
    fitted = SimpleNamespace(
        convergence_flag=0,
        params=pd.Series({"omega": 0.1, "alpha[1]": 0.2, "beta[1]": 0.8}),
        loglikelihood=-12.0,
        optimization_result=SimpleNamespace(message="converged"),
    )
    series = pd.Series(np.random.default_rng(1).normal(size=150))
    with patch("src.garch.arch_model") as model_factory:
        model_factory.return_value.fit.return_value = fitted
        with pytest.raises(GARCHFitError, match="inadmissible") as error:
            fit_garch(series)

    assert error.value.parameters.equals(fitted.params)
    assert error.value.diagnostics["converged"] is True
    assert error.value.diagnostics["covariance_stationarity"] is False
    assert error.value.diagnostics["admissible"] is False


def test_garch_rejects_nonconvergent_fit_and_preserves_estimates():
    fitted = SimpleNamespace(
        convergence_flag=1,
        params=pd.Series({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8}),
        loglikelihood=-12.0,
        optimization_result=SimpleNamespace(message="iteration limit reached"),
    )
    series = pd.Series(np.random.default_rng(2).normal(size=150))
    with patch("src.garch.arch_model") as model_factory:
        model_factory.return_value.fit.return_value = fitted
        with pytest.raises(GARCHFitError, match="did not converge") as error:
            fit_garch(series)

    assert error.value.parameters.equals(fitted.params)
    assert error.value.diagnostics["converged"] is False
    assert error.value.diagnostics["admissible"] is False


@pytest.mark.parametrize(
    "parameters",
    [
        {"omega": 0.0, "alpha[1]": 0.1, "beta[1]": 0.8},
        {"omega": 0.1, "alpha[1]": -0.1, "beta[1]": 0.8},
        {"omega": 0.1, "alpha[1]": 0.1, "beta[1]": -0.1},
    ],
)
def test_garch_rejects_nonpositive_variance_parameters(parameters):
    fitted = SimpleNamespace(
        convergence_flag=0,
        params=pd.Series(parameters),
        loglikelihood=-12.0,
        optimization_result=SimpleNamespace(message="converged"),
    )
    series = pd.Series(np.random.default_rng(3).normal(size=150))
    with patch("src.garch.arch_model") as model_factory:
        model_factory.return_value.fit.return_value = fitted
        with pytest.raises(GARCHFitError, match="inadmissible") as error:
            fit_garch(series)

    assert error.value.diagnostics["positive_parameters"] is False
    assert error.value.diagnostics["admissible"] is False
