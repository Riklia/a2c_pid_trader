# tests/unit/test_backtest_metrics.py
import numpy as np
import pandas as pd
import pytest

from backtest.metrics import total_return, sharpe_ratio, max_drawdown, vol_tracking_rmse


def test_total_return_flat_series_is_zero():
    equity = pd.Series([1.0, 1.0, 1.0])
    assert total_return(equity) == 0.0


def test_total_return_doubling_is_one():
    equity = pd.Series([1.0, 1.5, 2.0])
    assert total_return(equity) == pytest.approx(1.0)


def test_max_drawdown_flat_is_zero():
    equity = pd.Series([1.0, 1.0, 1.0])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_half_drop_is_half():
    equity = pd.Series([1.0, 2.0, 1.0])
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_sharpe_ratio_of_constant_positive_reward_diverges_to_nan():
    """Zero std -> Sharpe undefined, we return NaN, not inf."""
    rewards = pd.Series([0.001] * 100)
    assert np.isnan(sharpe_ratio(rewards, bars_per_year=8760))


def test_sharpe_ratio_scales_with_annualization_factor():
    rewards = pd.Series([0.01, -0.005, 0.008, -0.003, 0.006, -0.001] * 20)
    hourly = sharpe_ratio(rewards, bars_per_year=8760)
    daily = sharpe_ratio(rewards, bars_per_year=365)
    assert hourly == pytest.approx(daily * np.sqrt(24))


def test_vol_tracking_rmse_zero_when_realized_equals_target():
    vol = pd.Series([0.5, 0.5, 0.5])
    assert vol_tracking_rmse(vol, target_vol=0.5) == 0.0


def test_vol_tracking_rmse_matches_hand_calculation():
    # deviations from 0.5: -0.1, 0.0, +0.1
    vol = pd.Series([0.4, 0.5, 0.6])
    # RMSE = sqrt(((-0.1)^2 + 0^2 + 0.1^2) / 3) = sqrt(0.02/3)
    assert vol_tracking_rmse(vol, target_vol=0.5) == pytest.approx(np.sqrt(0.02 / 3))