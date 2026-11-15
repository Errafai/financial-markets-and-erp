"""Univariate GARCH(1,1) estimation and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class GARCHResult:
    """Validated outputs from one zero-mean GARCH fit."""

    parameters: pd.Series
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    diagnostics: dict[str, float | bool | str]
    fitted_model: object


class GARCHFitError(RuntimeError):
    """A failed or inadmissible GARCH fit with its estimated diagnostics."""

    def __init__(
        self,
        message: str,
        parameters: pd.Series,
        diagnostics: dict[str, float | bool | str],
    ) -> None:
        super().__init__(message)
        self.parameters = parameters.copy()
        self.diagnostics = diagnostics.copy()


def fit_garch(
    returns: pd.Series, *, p: int = 1, q: int = 1, rescale: bool = True
) -> GARCHResult:
    """Fit a zero-mean Gaussian GARCH(p,q) to percentage-scaled returns."""
    if (p, q) != (1, 1):
        raise ValueError("This project implements GARCH(1,1) only; use p=1 and q=1.")
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 100:
        raise ValueError("At least 100 valid return observations are required for GARCH.")
    if series.std(ddof=1) <= 0:
        raise ValueError("GARCH input must have non-zero variance.")
    model = arch_model(
        series * 100.0,
        mean="Zero",
        vol="GARCH",
        p=p,
        q=q,
        dist="normal",
        rescale=rescale,
    )
    fitted = model.fit(disp="off", show_warning=False)
    convergence = bool(fitted.convergence_flag == 0)
    params = fitted.params
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    finite_parameters = bool(np.isfinite([omega, alpha, beta]).all())
    positive_parameters = bool(
        finite_parameters and omega > 0 and alpha >= 0 and beta >= 0
    )
    persistence = alpha + beta if finite_parameters else float("nan")
    stationary = bool(finite_parameters and persistence < 1.0)
    admissible = convergence and positive_parameters and stationary
    diagnostics: dict[str, float | bool | str] = {
        "converged": convergence,
        "observations": float(len(series)),
        "log_likelihood": float(fitted.loglikelihood),
        "alpha_plus_beta": persistence,
        "positive_parameters": positive_parameters,
        "covariance_stationarity": stationary,
        "admissible": admissible,
        "optimizer_message": str(fitted.optimization_result.message),
    }
    if not convergence:
        raise GARCHFitError(
            f"GARCH optimizer did not converge: {fitted.optimization_result.message}",
            params,
            diagnostics,
        )
    if not admissible:
        raise GARCHFitError(
            "GARCH fit is inadmissible: require omega > 0, alpha >= 0, "
            "beta >= 0, and alpha + beta < 1.",
            params,
            diagnostics,
        )

    volatility = fitted.conditional_volatility / 100.0
    standardized = fitted.std_resid
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise RuntimeError("GARCH produced invalid conditional volatility.")
    if not np.isfinite(standardized).all():
        raise RuntimeError("GARCH produced non-finite standardized residuals.")
    standardized_values = standardized.dropna()
    ljung_box = acorr_ljungbox(standardized_values, lags=[10], return_df=True)
    squared_ljung_box = acorr_ljungbox(standardized_values**2, lags=[10], return_df=True)
    jb = jarque_bera(standardized_values)
    diagnostics.update(
        {
            "standardized_residual_ljung_box_pvalue_lag10": float(
                ljung_box["lb_pvalue"].iloc[0]
            ),
            "squared_residual_ljung_box_pvalue_lag10": float(
                squared_ljung_box["lb_pvalue"].iloc[0]
            ),
            "standardized_residual_jarque_bera_pvalue": float(jb.pvalue),
        }
    )
    return GARCHResult(params, volatility.rename(series.name), standardized.rename(series.name), diagnostics, fitted)


def fit_garch_panel(returns: pd.DataFrame) -> dict[str, GARCHResult]:
    """Estimate a separate GARCH(1,1) model for each column."""
    if returns.empty:
        raise ValueError("Returns panel must not be empty.")
    return {column: fit_garch(returns[column]) for column in returns.columns}


def align_standardized_residuals(
    results: dict[str, GARCHResult],
) -> pd.DataFrame:
    """Inner-align admissible GARCH residuals without filling or interpolating.

    The returned index contains only dates present with finite residuals for
    every fitted series, preserving the original observation dates.
    """
    if len(results) < 2:
        raise ValueError("At least two fitted GARCH series are required for DCC.")

    residual_series: dict[str, pd.Series] = {}
    for name, result in results.items():
        if not result.diagnostics.get("converged", False):
            raise ValueError(f"GARCH series {name!r} did not converge.")
        if not result.diagnostics.get("admissible", False):
            raise ValueError(f"GARCH series {name!r} is not admissible for DCC.")
        residuals = result.standardized_residuals
        if residuals.index.has_duplicates:
            raise ValueError(f"GARCH residual dates for {name!r} are not unique.")
        if not residuals.index.is_monotonic_increasing:
            raise ValueError(f"GARCH residual dates for {name!r} are not sorted.")
        residual_series[name] = residuals.replace([np.inf, -np.inf], np.nan)

    aligned = pd.concat(residual_series, axis=1, join="inner").dropna(how="any")
    if len(aligned) < 30:
        raise ValueError(
            "At least 30 common finite residual dates are required for DCC-GARCH."
        )
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Aligned standardized residuals contain non-finite values.")
    return aligned
