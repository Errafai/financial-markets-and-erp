"""Normalize the five manually downloaded Moroccan bank CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"
REAL_LABEL = "REAL MARKET DATA - FROM LOCAL CSV IMPORTS"
REAL_BANKS = ("ATW", "BCP", "BOA", "CIH", "CDM")


def load_real_bank_prices(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    series: dict[str, pd.Series] = {}

    for bank in REAL_BANKS:
        candidates = sorted(raw_dir.glob(f"*{bank}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No real CSV found for {bank} in {raw_dir}.")

        frame = pd.read_csv(
            candidates[0],
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
        date_column = next(
            (column for column in frame.columns if column.strip().lower() in {"session", "date"}),
            None,
        )
        close_column = next(
            (column for column in frame.columns if column.strip().lower() in {"last price", "close"}),
            None,
        )
        if date_column is None or close_column is None:
            raise ValueError(f"{candidates[0]} must contain Session/Date and Last Price/Close columns.")

        dates = pd.to_datetime(frame[date_column], dayfirst=True, errors="raise")
        close = pd.to_numeric(frame[close_column], errors="raise")
        if dates.hasnans or close.isna().any():
            raise ValueError(f"{candidates[0]} contains missing dates or closing prices.")
        if (close <= 0).any():
            raise ValueError(f"{candidates[0]} contains non-positive closing prices.")
        series[bank] = pd.Series(close.to_numpy(), index=pd.DatetimeIndex(dates, name="date"))

    prices = pd.concat(series, axis=1).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Real bank CSVs contain duplicate dates.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Real bank dates must be sorted.")
    if prices.isna().any().any():
        raise ValueError("Real bank closing prices are not aligned across all five banks.")
    if len(prices) < 100:
        raise ValueError("At least 100 aligned real observations are required.")
    return prices


def save_real_dataset(
    raw_dir: Path,
    output_dir: Path = REAL_DATA_DIR,
) -> tuple[Path, Path, Path, Path]:
    prices = load_real_bank_prices(raw_dir)
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    returns.columns = list(REAL_BANKS)
    market = pd.DataFrame({"MASI_REAL": returns.mean(axis=1)})

    combined = pd.concat(
        [
            prices.add_prefix("price_"),
            returns.add_prefix("return_"),
            market,
            pd.Series(False, index=returns.index, name="stress_period"),
        ],
        axis=1,
    ).dropna(how="any")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "bank_prices.csv",
        output_dir / "bank_returns.csv",
        output_dir / "market_index.csv",
        output_dir / "financial_dataset.csv",
    ]
    for frame, path in zip(
        (prices, returns, market, combined),
        targets,
    ):
        labeled = frame.copy()
        labeled["data_status"] = REAL_LABEL
        labeled.to_csv(path, index_label="date")
    return tuple(targets)


def load_real_dataset(
    data_dir: Path = REAL_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = [
        data_dir / "bank_prices.csv",
        data_dir / "bank_returns.csv",
        data_dir / "market_index.csv",
        data_dir / "financial_dataset.csv",
    ]
    if not all(path.exists() for path in targets):
        raise FileNotFoundError("Real canonical dataset is incomplete or missing.")

    frames = tuple(pd.read_csv(path, parse_dates=["date"], index_col="date") for path in targets)
    for frame in frames:
        if "data_status" not in frame or not frame["data_status"].eq(REAL_LABEL).all():
            raise ValueError("Canonical real CSV is missing the required real-data label.")
        frame.drop(columns="data_status", inplace=True)
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Canonical real CSV contains non-finite numeric values.")
    if not all(frame.index.is_monotonic_increasing for frame in frames):
        raise ValueError("Real CSV dates must be sorted.")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REAL_DATA_DIR)
    args = parser.parse_args()
    paths = save_real_dataset(args.raw_dir, args.output_dir)
    print(" ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
