"""DCC-GARCH(1,1) estimation on univariate-GARCH standardized residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DCCResult:
    """DCC estimates and conditional correlation matrices."""

    a: float
    b: float
    correlations: np.ndarray
    average_correlation: pd.DataFrame
    converged: bool
    log_likelihood: float
    observation_dates: pd.Index
    columns: list[str]


def _validate_parameters(a: float, b: float) -> None:
    if not np.isfinite([a, b]).all() or a < 0 or b < 0 or a + b >= 1:
        raise ValueError("DCC parameters must satisfy a >= 0, b >= 0, and a + b < 1.")


def dcc_filter(
    standardized_residuals: pd.DataFrame | np.ndarray,
    a: float,
    b: float,
    *,
    check_positive_definite: bool = True,
) -> np.ndarray:
    """Apply the DCC recursion and return one correlation matrix per row.

    Input columns must be standardized residual series from univariate GARCH fits.
    """
    _validate_parameters(a, b)
    values = np.asarray(standardized_residuals, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("DCC input must be a 2-D matrix with at least 2 rows and columns.")
    if not np.isfinite(values).all():
        raise ValueError("DCC input contains non-finite standardized residuals.")
    if (values.std(axis=0, ddof=1) <= 0).any():
        raise ValueError("Every DCC series must have positive variance.")
    # Standardized residuals are supplied by the univariate GARCH stage.
    z = values
    q_bar = z.T @ z / len(z)
    q_bar = (q_bar + q_bar.T) / 2.0
    if not np.isfinite(q_bar).all():
        raise ValueError("Could not calculate a finite unconditional correlation matrix.")
    q = q_bar.copy()
    correlations = np.empty((len(z), z.shape[1], z.shape[1]), dtype=float)
    for t in range(len(z)):
        if t:
            q = (1.0 - a - b) * q_bar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        diagonal = np.diag(q)
        if (diagonal <= 0).any() or not np.isfinite(q).all():
            raise RuntimeError(f"Invalid DCC covariance state at observation {t}.")
        inv_sqrt = np.diag(1.0 / np.sqrt(diagonal))
        r = inv_sqrt @ q @ inv_sqrt
        r = (r + r.T) / 2.0
        if check_positive_definite and np.linalg.eigvalsh(r).min() <= 0:
            raise RuntimeError(f"DCC correlation matrix is not positive definite at {t}.")
        correlations[t] = r
    return correlations


def fit_dcc(
    standardized_residuals: pd.DataFrame,
    *,
    initial: tuple[float, float] = (0.03, 0.94),
    maxiter: int = 300,
) -> DCCResult:
    """Estimate DCC parameters by constrained Gaussian quasi-maximum likelihood."""
    if not isinstance(standardized_residuals, pd.DataFrame):
        raise TypeError("DCC estimation requires a DataFrame with dated residual columns.")
    if standardized_residuals.columns.has_duplicates:
        raise ValueError("DCC residual column names must be unique.")
    if not standardized_residuals.index.is_unique:
        raise ValueError("DCC residual observation indices must be unique.")
    if not standardized_residuals.index.is_monotonic_increasing:
        raise ValueError("DCC residual observations must be sorted chronologically.")
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    values = np.asarray(standardized_residuals, dtype=float)
    if values.ndim != 2 or len(values) < 30 or values.shape[1] < 2:
        raise ValueError("DCC estimation requires >=30 rows and >=2 residual series.")
    if not np.isfinite(values).all():
        raise ValueError("Standardized residuals must be finite.")
    if (values.std(axis=0, ddof=1) <= 0).any():
        raise ValueError("Every DCC series must have positive variance.")
    _validate_parameters(*initial)

    def objective(params: np.ndarray) -> float:
        a, b = params
        try:
            matrices = dcc_filter(
                values,
                float(a),
                float(b),
                check_positive_definite=False,
            )
            z = values
            total = 0.0
            for row, corr in zip(z, matrices):
                sign, logdet = np.linalg.slogdet(corr)
                if sign <= 0:
                    return 1e100
                total += logdet + row @ np.linalg.solve(corr, row) - row @ row
            return float(total)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
            return 1e100

    optimized = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="SLSQP",
        bounds=((0.0, 0.999), (0.0, 0.999)),
        constraints=({"type": "ineq", "fun": lambda x: 0.999 - x[0] - x[1]},),
        options={"maxiter": maxiter, "ftol": 1e-8},
    )
    a, b = map(float, optimized.x)
    _validate_parameters(a, b)
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError(f"DCC optimizer failed: {optimized.message}")
    matrices = dcc_filter(standardized_residuals, a, b)
    names = list(standardized_residuals.columns)
    avg = pd.DataFrame(matrices.mean(axis=0), index=names, columns=names)
    dates = standardized_residuals.index
    return DCCResult(
        a,
        b,
        matrices,
        avg,
        True,
        -0.5 * float(optimized.fun),
        dates,
        names,
    )


def upper_triangle_features(correlation_matrices: np.ndarray) -> np.ndarray:
    """Extract unique off-diagonal correlations from a sequence of matrices."""
    matrices = np.asarray(correlation_matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Correlation matrices must have shape (time, assets, assets).")
    if matrices.shape[1] < 2 or not np.isfinite(matrices).all():
        raise ValueError("Matrices must contain at least two assets and finite values.")
    indices = np.triu_indices(matrices.shape[1], k=1)
    return matrices[:, indices[0], indices[1]]
