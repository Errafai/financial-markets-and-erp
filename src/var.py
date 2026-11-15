"""Historical and Gaussian parametric Value-at-Risk in positive loss units."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5 (e.g. 0.05).")


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Return the negative linear-interpolated lower-tail return quantile.

    VaR is expressed in loss units. It is positive when the selected return
    quantile is a loss, but can be negative when even that quantile is a gain.
    """
    _validate_alpha(alpha)
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("At least 20 finite return observations are required.")
    return float(-np.quantile(values, alpha))


def parametric_var(
    returns: pd.Series, alpha: float = 0.05, *, mean: float | None = None, std: float | None = None
) -> float:
    """Estimate Gaussian positive-loss VaR from a return series or supplied moments."""
    _validate_alpha(alpha)
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite returns are required.")
    mu = float(np.mean(values) if mean is None else mean)
    sigma = float(np.std(values, ddof=1) if std is None else std)
    has_observed_variation = bool(np.ptp(values) > 0)
    if (
        not np.isfinite([mu, sigma]).all()
        or sigma <= 0
        or (std is None and not has_observed_variation)
    ):
        raise ValueError("Mean must be finite and standard deviation must be positive.")
    return float(-(mu + norm.ppf(alpha) * sigma))
