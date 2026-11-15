"""Date alignment and return calculations for synthetic price series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort observations, remove duplicate dates (keeping last), and validate prices."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Prices must use a DatetimeIndex.")
    cleaned = prices.copy().sort_index()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    if cleaned.empty or cleaned.shape[1] == 0:
        raise ValueError("At least one non-empty price series is required.")
    if (cleaned <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    if cleaned.empty or (cleaned.notna().sum() < 2).any():
        raise ValueError("Every price series must contain at least two valid observations.")
    # Keep dates where all series are missing so a later return cannot bridge
    # across a non-observation after cleaning.
    return cleaned


def calculate_log_returns(
    prices: pd.DataFrame, *, align: str = "inner", drop_missing: bool = True
) -> pd.DataFrame:
    """Calculate log(P_t/P_(t-1)); align series on common dates by default."""
    if align not in {"inner", "outer"}:
        raise ValueError("align must be either 'inner' or 'outer'.")
    cleaned = clean_price_data(prices)
    returns = np.log(cleaned / cleaned.shift(1))
    if drop_missing:
        returns = returns.dropna(how="any" if align == "inner" else "all")
    if returns.empty or not returns.notna().any().any():
        raise ValueError("No valid aligned returns could be calculated from the prices.")
    return returns.replace([np.inf, -np.inf], np.nan)
