"""Plotting helpers with explicit synthetic-data labeling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REAL_LABEL


def plot_series(
    series: pd.Series | pd.DataFrame, title: str, ylabel: str, output: Path | None = None
) -> plt.Figure:
    """Plot one or more time series and clearly mark them as simulated."""
    fig, axis = plt.subplots(figsize=(12, 5))
    series.plot(ax=axis)
    axis.set_title(f"{title}\n{REAL_LABEL}", fontsize=11)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Date")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation(matrix: pd.DataFrame, output: Path | None = None) -> plt.Figure:
    """Draw a labeled correlation heatmap."""
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap="vlag", vmin=-1, vmax=1, center=0, ax=axis)
    axis.set_title(f"Average DCC correlation — {REAL_LABEL}")
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig
