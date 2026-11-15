"""System-wide dynamic-correlation summaries and CoVaR risk ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.covar import CoVaRResult


def rank_systemic_risk(covar_results: dict[str, CoVaRResult]) -> pd.DataFrame:
    """Rank institutions by most negative ΔCoVaR return estimate.

    This is a ranking of the estimated downside association, not a causal
    decomposition of each institution's contribution to systemic risk.
    """
    rows = [
        {
            "bank_code": bank,
            "delta_covar": float(result.delta_covar),
            "covar_distress": float(result.covar_distress),
            "covar_normal": float(result.covar_normal),
        }
        for bank, result in covar_results.items()
    ]
    if not rows:
        raise ValueError("At least one CoVaR result is required.")
    return pd.DataFrame(rows).sort_values("delta_covar").reset_index(drop=True)


def correlation_summary(
    matrices: np.ndarray, dates: pd.Index, bank_names: list[str]
) -> pd.DataFrame:
    """Summarize each correlation matrix by its mean off-diagonal correlation."""
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Expected correlation matrices with shape (time, n, n).")
    if len(dates) != len(matrices) or len(bank_names) != matrices.shape[1]:
        raise ValueError("Dates, matrices, and bank names must have matching dimensions.")
    if matrices.shape[1] < 2 or not np.isfinite(matrices).all():
        raise ValueError("Need finite correlation matrices for at least two banks.")
    if not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-8):
        raise ValueError("Every dynamic correlation matrix must be symmetric.")
    if not np.allclose(
        np.diagonal(matrices, axis1=1, axis2=2), 1.0, atol=1e-8
    ):
        raise ValueError("Every dynamic correlation matrix must have a unit diagonal.")
    if (np.abs(matrices) > 1.0 + 1e-8).any():
        raise ValueError("Correlation values must be between -1 and 1.")
    i, j = np.triu_indices(matrices.shape[1], k=1)
    average = matrices[:, i, j].mean(axis=1)
    return pd.DataFrame({"average_correlation": average}, index=dates)
