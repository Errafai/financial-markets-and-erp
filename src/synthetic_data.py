"""Generate a reproducible, synthetic-only daily financial dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BANKS,
    MARKET_COLUMN,
    RANDOM_SEED,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_END_DATE,
    SYNTHETIC_START_DATE,
    SYNTHETIC_LABEL,
)

_VOLATILITY = np.array([0.010, 0.013, 0.012, 0.015, 0.014])
_OMEGA = _VOLATILITY**2 * (1.0 - 0.08 - 0.88)
_ALPHA = np.full(len(BANKS), 0.08)
_BETA = np.full(len(BANKS), 0.88)


def generate_synthetic_data(
    seed: int = RANDOM_SEED,
    start: str = SYNTHETIC_START_DATE,
    end: str = SYNTHETIC_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return prices, bank returns, market index, and a combined wide dataset.

    Returns follow correlated GARCH(1,1)-like recursions. Two deterministic
    stress windows increase conditional variance and common-factor dependence.
    No real market observations or external data sources are used.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end, name="date")
    bank_codes = list(BANKS)
    n_obs, n_banks = len(dates), len(bank_codes)
    if n_obs < 300:
        raise ValueError("At least 300 business-day observations are required.")

    base_corr = np.array(
        [
            [1.00, 0.42, 0.38, 0.32, 0.35],
            [0.42, 1.00, 0.40, 0.34, 0.36],
            [0.38, 0.40, 1.00, 0.37, 0.39],
            [0.32, 0.34, 0.37, 1.00, 0.41],
            [0.35, 0.36, 0.39, 0.41, 1.00],
        ]
    )
    stress = np.zeros(n_obs, dtype=bool)
    stress[max(0, int(n_obs * 0.42)) : int(n_obs * 0.47)] = True
    stress[int(n_obs * 0.78) : int(n_obs * 0.84)] = True
    common = rng.standard_normal(n_obs)
    independent = rng.standard_normal((n_obs, n_banks))
    innovations = np.empty((n_obs, n_banks), dtype=float)
    for t in range(n_obs):
        rho_common = 0.24 if not stress[t] else 0.68
        base = independent[t] @ np.linalg.cholesky(base_corr).T
        innovations[t] = np.sqrt(1.0 - rho_common**2) * base + rho_common * common[t]

    variance = np.empty((n_obs, n_banks), dtype=float)
    returns = np.empty_like(variance)
    variance[0] = _VOLATILITY**2
    for t in range(n_obs):
        if t:
            regime_omega = _OMEGA * (4.0 if stress[t] else 1.0)
            variance[t] = (
                regime_omega
                + _ALPHA * returns[t - 1] ** 2
                + _BETA * variance[t - 1]
            )
        returns[t] = np.sqrt(variance[t]) * innovations[t]

    returns_frame = pd.DataFrame(returns, index=dates, columns=bank_codes)
    returns_frame.index.name = "date"
    prices = 100.0 * np.exp(returns_frame.cumsum())
    prices.index.name = "date"

    market_returns = returns_frame.mean(axis=1)
    market_index = pd.DataFrame(
        {MARKET_COLUMN: 1000.0 * np.exp(market_returns.cumsum())},
        index=dates,
    )
    market_index.index.name = "date"
    combined = pd.concat(
        [
            prices.add_prefix("price_"),
            returns_frame.add_prefix("return_"),
            market_index,
            pd.Series(stress, index=dates, name="stress_period"),
        ],
        axis=1,
    )
    combined["data_status"] = SYNTHETIC_LABEL
    combined.index.name = "date"

    for frame in (prices, returns_frame, market_index, combined):
        frame.attrs["data_status"] = SYNTHETIC_LABEL

    return prices, returns_frame, market_index, combined


def save_synthetic_data(
    output_dir: Path = SYNTHETIC_DATA_DIR, *, force: bool = False, seed: int = RANDOM_SEED
) -> list[Path]:
    """Write the canonical CSVs without replacing a complete or partial dataset.

    Set ``force=True`` only when intentionally regenerating and replacing the
    complete set of canonical files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "bank_prices.csv",
        output_dir / "bank_returns.csv",
        output_dir / "market_index.csv",
        output_dir / "financial_dataset.csv",
    ]
    existing = [path for path in targets if path.exists()]
    if not force:
        if len(existing) == len(targets):
            return targets
        if existing:
            missing = [path.name for path in targets if not path.exists()]
            raise FileExistsError(
                "Refusing to regenerate an incomplete synthetic dataset. "
                f"Existing files were preserved; missing files: {missing}. "
                "Use force=True only to intentionally replace the complete dataset."
            )

    prices, returns, market, combined = generate_synthetic_data(seed=seed)
    for frame, path in zip((prices, returns, market, combined), targets):
        labeled = frame.copy()
        if path.name != "financial_dataset.csv":
            labeled["data_status"] = SYNTHETIC_LABEL
        labeled.to_csv(path, index_label="date")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally regenerate and replace all canonical CSV files.",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Generate the initial dataset only when all canonical CSV files are absent.",
    )
    args = parser.parse_args()
    paths = save_synthetic_data(
        args.output_dir,
        force=args.force,
        seed=args.seed,
    )
    for path in paths:
        print(f"Dataset ready: {path} ({SYNTHETIC_LABEL})")


if __name__ == "__main__":
    main()
