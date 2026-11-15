"""Static and rolling quantile-regression CoVaR calculations."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning


@dataclass
class CoVaRResult:
    """CoVaR in return units; downside estimates are commonly negative."""

    quantile: float
    bank_distress_return: float
    bank_normal_return: float
    covar_distress: float
    covar_normal: float
    delta_covar: float
    intercept: float
    slope: float
    observations: int
    iterations: int = 0
    solver: str = "statsmodels-irls"


def leave_one_out_system_return(
    bank_returns: pd.DataFrame, bank_code: str
) -> pd.Series:
    """Build an equal-weight peer-system return excluding the target institution.

    Exclusion avoids the direct part-whole relationship that would arise if a
    bank's return were included in the system return regressed on that bank.
    """
    if bank_code not in bank_returns.columns:
        raise KeyError(f"Unknown institution code: {bank_code}")
    peers = bank_returns.drop(columns=bank_code)
    if peers.shape[1] < 1:
        raise ValueError("At least one peer institution is required.")
    peers = peers.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    system_return = peers.mean(axis=1, skipna=False)
    if system_return.dropna().empty:
        raise ValueError("No complete peer-system return observations are available.")
    return system_return.rename("peer_system_return")


def _paired_returns(system_returns: pd.Series, bank_returns: pd.Series) -> pd.DataFrame:
    if system_returns.index.has_duplicates or bank_returns.index.has_duplicates:
        raise ValueError("Return series must not contain duplicate observation indices.")
    pair = pd.concat(
        [system_returns.rename("system"), bank_returns.rename("bank")], axis=1, join="inner"
    )
    pair = pair.apply(pd.to_numeric, errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 50:
        raise ValueError("At least 50 paired finite return observations are required.")
    if pair["bank"].std(ddof=1) <= 0 or pair["system"].std(ddof=1) <= 0:
        raise ValueError("Paired return series must both have non-zero variance.")
    return pair.sort_index()


def estimate_covar(
    system_returns: pd.Series,
    bank_returns: pd.Series,
    quantile: float = 0.05,
) -> CoVaRResult:
    """Fit X_system = alpha_q + gamma_q * X_bank + error at quantile q.

    ΔCoVaR is distress-state CoVaR minus median-state CoVaR, both in return
    units. Negative ΔCoVaR indicates a worse system-tail return under distress.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5.")
    pair = _paired_returns(system_returns, bank_returns)
    design = sm.add_constant(pair["bank"], has_constant="add")
    quantile_model = sm.QuantReg(pair["system"], design)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IterationLimitWarning)
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            fitted = quantile_model.fit(
                q=quantile, max_iter=10000, p_tol=1e-7
            )
            coefficients = fitted.params[["const", "bank"]].to_numpy(dtype=float)
            iterations = int(fitted.iterations)
            solver = "statsmodels-irls"
        except (IterationLimitWarning, ConvergenceWarning):
            # Linear programming minimizes the same quantile (pinball) loss
            # without the IRLS iteration cycle/limit failure.
            exog = design.to_numpy(dtype=float)
            endog = pair["system"].to_numpy(dtype=float)
            n_observations, n_parameters = exog.shape
            objective = np.concatenate(
                [
                    np.zeros(n_parameters),
                    np.full(n_observations, quantile),
                    np.full(n_observations, 1.0 - quantile),
                ]
            )
            equality = np.column_stack(
                [
                    exog,
                    np.eye(n_observations),
                    -np.eye(n_observations),
                ]
            )
            optimized = linprog(
                objective,
                A_eq=equality,
                b_eq=endog,
                bounds=[(None, None)] * n_parameters
                + [(0.0, None)] * (2 * n_observations),
                method="highs",
            )
            if not optimized.success or optimized.x is None:
                raise RuntimeError(
                    "Quantile regression failed to converge with both "
                    f"statsmodels IRLS and SciPy HiGHS: {optimized.message}"
                )
            coefficients = optimized.x[:n_parameters]
            iterations = int(optimized.nit or 0)
            solver = "scipy-highs-linear-program"
    intercept, slope = map(float, coefficients)
    if not np.isfinite([intercept, slope]).all():
        raise RuntimeError("Quantile regression returned non-finite coefficients.")
    distressed = float(pair["bank"].quantile(quantile))
    normal = float(pair["bank"].median())
    covar_distress = intercept + slope * distressed
    covar_normal = intercept + slope * normal
    return CoVaRResult(
        quantile,
        distressed,
        normal,
        float(covar_distress),
        float(covar_normal),
        float(covar_distress - covar_normal),
        intercept,
        slope,
        len(pair),
        iterations,
        solver,
    )


def rolling_covar(
    system_returns: pd.Series,
    bank_returns: pd.Series,
    *,
    quantile: float = 0.05,
    window: int = 250,
    step: int = 1,
) -> pd.DataFrame:
    """Estimate date-specific CoVaR using only trailing observations."""
    if window < 50:
        raise ValueError("Rolling window must be at least 50 observations.")
    if step < 1:
        raise ValueError("step must be a positive integer.")
    pair = _paired_returns(system_returns, bank_returns)
    if len(pair) < window:
        raise ValueError("The aligned return history is shorter than the rolling window.")
    rows = []
    endpoints = list(range(window, len(pair) + 1, step))
    if endpoints[-1] != len(pair):
        endpoints.append(len(pair))
    for end in endpoints:
        subset = pair.iloc[end - window : end]
        result = estimate_covar(subset["system"], subset["bank"], quantile)
        rows.append(
            {
                "date": subset.index[-1],
                "covar_distress": result.covar_distress,
                "covar_normal": result.covar_normal,
                "delta_covar": result.delta_covar,
            }
        )
    return pd.DataFrame(rows).set_index("date")
