import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from src.covar import (
    estimate_covar,
    leave_one_out_system_return,
    rolling_covar,
)


def test_quantile_regression_covar_and_delta_are_consistent():
    rng = np.random.default_rng(9)
    bank = rng.normal(0, 0.02, 600)
    system = 0.4 * bank + rng.normal(0, 0.015, 600)
    index = pd.bdate_range("2020-01-01", periods=len(bank))
    result = estimate_covar(pd.Series(system, index=index), pd.Series(bank, index=index))
    assert result.covar_distress < result.covar_normal
    assert np.isclose(result.delta_covar, result.covar_distress - result.covar_normal)
    assert result.observations == 600


def test_rolling_covar_uses_trailing_window_dates():
    rng = np.random.default_rng(2)
    bank = rng.normal(0, 0.02, 100)
    system = bank * 0.3 + rng.normal(0, 0.02, 100)
    dates = pd.bdate_range("2024-01-01", periods=100)
    result = rolling_covar(pd.Series(system, dates), pd.Series(bank, dates), window=60)
    assert len(result) == 41
    assert result.index[0] == dates[59]


def test_quantile_regression_uses_linear_program_after_irls_nonconvergence(
    monkeypatch,
):
    rng = np.random.default_rng(2)
    bank = rng.normal(0, 0.02, 60)
    system = bank * 0.3 + rng.normal(0, 0.02, 60)
    dates = pd.bdate_range("2024-01-01", periods=60)

    def fail_irls(*args, **kwargs):
        raise IterationLimitWarning("test IRLS non-convergence")

    monkeypatch.setattr("src.covar.sm.QuantReg.fit", fail_irls)
    result = estimate_covar(
        pd.Series(system, index=dates),
        pd.Series(bank, index=dates),
        quantile=0.05,
    )

    assert result.solver == "scipy-highs-linear-program"
    assert np.isfinite(
        [
            result.intercept,
            result.slope,
            result.covar_distress,
            result.covar_normal,
            result.delta_covar,
        ]
    ).all()
    assert np.isclose(
        result.delta_covar, result.covar_distress - result.covar_normal
    )


def test_covar_rejects_invalid_quantile_and_short_sample():
    series = pd.Series(np.random.default_rng(1).normal(size=100))
    with pytest.raises(ValueError, match="quantile"):
        estimate_covar(series, series, quantile=0.7)
    with pytest.raises(ValueError, match="50"):
        estimate_covar(series.iloc[:20], series.iloc[:20])


def test_leave_one_out_proxy_excludes_target_bank():
    returns = pd.DataFrame(
        {
            "target": [0.10, 0.20],
            "peer_a": [0.02, 0.04],
            "peer_b": [0.04, 0.08],
        }
    )
    proxy = leave_one_out_system_return(returns, "target")
    assert np.allclose(proxy, [0.03, 0.06])


def test_covar_rejects_duplicate_indices_and_constant_series():
    rng = np.random.default_rng(12)
    values = rng.normal(size=60)
    duplicate_dates = pd.Series(values, index=[1] * 60)
    regular = pd.Series(values)
    with pytest.raises(ValueError, match="duplicate"):
        estimate_covar(duplicate_dates, regular)
    with pytest.raises(ValueError, match="non-zero variance"):
        estimate_covar(pd.Series(np.ones(60)), regular)


def test_rolling_covar_first_output_does_not_depend_on_future_observations():
    rng = np.random.default_rng(42)
    bank = pd.Series(
        rng.normal(size=80) * 0.02,
        index=pd.bdate_range("2024-01-01", periods=80),
    )
    system = 0.3 * bank + pd.Series(rng.normal(size=80) * 0.015, index=bank.index)
    next_date = bank.index[-1] + pd.offsets.BDay()
    extended_bank = pd.concat([bank, pd.Series([50.0], index=[next_date])])
    extended_system = pd.concat([system, pd.Series([-50.0], index=[next_date])])
    original = rolling_covar(system, bank, window=60)
    extended = rolling_covar(extended_system, extended_bank, window=60)
    assert np.isclose(original.iloc[0]["delta_covar"], extended.iloc[0]["delta_covar"])
