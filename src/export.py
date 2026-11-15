"""Export labeled synthetic analysis results in CSV and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import SYNTHETIC_LABEL


def export_indicators(
    indicators: pd.DataFrame, output_dir: Path, *, stem: str = "synthetic_financial_results"
) -> tuple[Path, Path]:
    """Export synthetic analysis results to CSV and JSON with explicit provenance."""
    if indicators.empty:
        raise ValueError("Cannot export an empty indicators table.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = indicators.copy()
    if frame.index.name or isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
    frame["data_status"] = SYNTHETIC_LABEL
    for column in frame.columns:
        if column.lower() in {"date", "datetime"} and pd.api.types.is_datetime64_any_dtype(
            frame[column]
        ):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d")
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path
