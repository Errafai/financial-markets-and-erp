"""Load canonical synthetic datasets and validate imported demonstration CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BANKS, SYNTHETIC_DATA_DIR, SYNTHETIC_LABEL
from src.real_data import load_real_dataset
from src.synthetic_data import save_synthetic_data


def load_synthetic_dataset(
    data_dir: Path = SYNTHETIC_DATA_DIR, *, generate_if_missing: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four canonical CSVs without silently regenerating existing data."""
    files = [
        data_dir / "bank_prices.csv",
        data_dir / "bank_returns.csv",
        data_dir / "market_index.csv",
        data_dir / "financial_dataset.csv",
    ]
    existing = [path for path in files if path.exists()]
    if not existing:
        if not generate_if_missing:
            raise FileNotFoundError(
                f"Canonical synthetic dataset is absent from {data_dir}."
            )
        save_synthetic_data(data_dir)
    elif len(existing) != len(files):
        missing = [path.name for path in files if not path.exists()]
        raise FileNotFoundError(
            "Canonical synthetic dataset is incomplete; refusing automatic "
            f"regeneration to protect existing files. Missing: {missing}."
        )
    loaded = tuple(pd.read_csv(path, parse_dates=["date"], index_col="date") for path in files)
    for path, frame in zip(files[:3], loaded[:3]):
        if "data_status" not in frame or not frame["data_status"].eq(SYNTHETIC_LABEL).all():
            raise ValueError(f"{path.name} is missing the required synthetic-data label.")
    if "data_status" not in loaded[3] or not loaded[3]["data_status"].eq(SYNTHETIC_LABEL).all():
        raise ValueError("financial_dataset.csv is missing the required synthetic-data label.")
    frames = tuple(frame.drop(columns="data_status") for frame in loaded)
    for frame in frames:
        frame.attrs["data_status"] = SYNTHETIC_LABEL
    for path, frame in zip(files, frames):
        try:
            values = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path.name} contains non-numeric dataset fields.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} contains missing or non-finite dataset values.")
    if not all(frame.index.is_monotonic_increasing for frame in frames):
        raise ValueError("Synthetic CSV dates must be sorted in ascending order.")
    if not all(frame.index.is_unique for frame in frames):
        raise ValueError("Synthetic CSV files contain duplicate dates.")
    if any(not frame.index.equals(frames[0].index) for frame in frames[1:]):
        raise ValueError("Canonical synthetic CSV files do not have aligned dates.")
    if frames[0].empty or not set(BANKS).issubset(frames[0].columns):
        raise ValueError("Bank price CSV is empty or missing expected simulated banks.")
    if (frames[0].loc[:, list(BANKS)] <= 0).any().any():
        raise ValueError("Simulated bank prices must be strictly positive.")
    if not set(BANKS).issubset(frames[1].columns):
        raise ValueError("Bank return CSV is missing expected simulated institutions.")
    if "MASI_SIM" not in frames[2].columns:
        raise ValueError("Market-index CSV is missing MASI_SIM.")
    if not set(BANKS).issubset(
        {column.removeprefix("price_") for column in frames[3].columns if column.startswith("price_")}
    ):
        raise ValueError("Combined synthetic dataset is missing simulated bank prices.")
    if "stress_period" not in frames[3].columns:
        raise ValueError("Combined synthetic dataset is missing the stress-period flag.")
    return frames  # type: ignore[return-value]


def load_market_dataset(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the repository's real canonical market dataset."""
    from src.config import REAL_DATA_DIR

    return load_real_dataset(REAL_DATA_DIR if data_dir is None else data_dir)


def load_returns_csv(path: Path) -> pd.DataFrame:
    """Load a user-provided returns CSV for academic demonstrations only."""
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError("CSV must include a 'date' column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("CSV contains missing observation dates.")
    frame = frame.set_index("date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("CSV contains duplicate dates; resolve them before analysis.")
    if "data_status" not in frame.columns or not frame["data_status"].eq(SYNTHETIC_LABEL).all():
        raise ValueError(
            "CSV must explicitly label every row as the project's synthetic data."
        )
    return_columns = frame.drop(columns=["data_status"])
    if return_columns.empty:
        raise ValueError("CSV must contain at least one numeric return series.")
    numeric = return_columns.apply(pd.to_numeric, errors="coerce")
    if (numeric.isna() & return_columns.notna()).any().any():
        raise ValueError("CSV return series must contain only numeric values.")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("CSV return series must contain finite values.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.empty or numeric.isna().all().any():
        raise ValueError("CSV must contain at least one populated numeric return series.")
    numeric.attrs["data_status"] = SYNTHETIC_LABEL
    return numeric
