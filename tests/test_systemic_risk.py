import numpy as np
import pandas as pd

from src.covar import CoVaRResult
from src.systemic_risk import correlation_summary, rank_systemic_risk


def test_risk_ranking_places_largest_downside_first():
    results = {
        "BANK_A": CoVaRResult(0.05, -0.03, 0, -0.04, -0.01, -0.03, 0, 1, 100),
        "BANK_B": CoVaRResult(0.05, -0.02, 0, -0.02, -0.01, -0.01, 0, 1, 100),
    }
    ranking = rank_systemic_risk(results)
    assert ranking.iloc[0]["bank_code"] == "BANK_A"


def test_correlation_summary_averages_unique_off_diagonal_pairs():
    rng = np.random.default_rng(15)
    dates = pd.bdate_range("2025-01-01", periods=20)
    matrices = np.repeat(np.eye(3)[None, :, :], len(dates), axis=0)
    matrices[:, 0, 1] = matrices[:, 1, 0] = rng.uniform(-0.2, 0.8, len(dates))
    matrices[:, 0, 2] = matrices[:, 2, 0] = 0.2
    matrices[:, 1, 2] = matrices[:, 2, 1] = 0.4
    summary = correlation_summary(matrices, dates, ["a", "b", "c"])
    assert np.allclose(
        summary["average_correlation"],
        (matrices[:, 0, 1] + 0.2 + 0.4) / 3,
    )


def test_correlation_summary_rejects_mismatched_metadata():
    matrices = np.repeat(np.eye(2)[None, :, :], 5, axis=0)
    with np.testing.assert_raises_regex(ValueError, "matching dimensions"):
        correlation_summary(matrices, pd.Index(range(4)), ["a", "b"])
