import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    total_return, sharpe_ratio, max_drawdown, vol_tracking_rmse,
    annualized_vol, rolling_portfolio_vol_rmse, turnover, beta_alpha,
)


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

def test_annualized_vol_scales_with_sqrt_of_bars():
    rewards = pd.Series([0.01, -0.01] * 50)
    assert annualized_vol(rewards, bars_per_year=100) == pytest.approx(rewards.std() * 10)


def test_rolling_portfolio_vol_rmse_zero_when_rolling_vol_equals_target():
    rewards = pd.Series([0.01, -0.01] * 50)
    window_vol = pd.Series([0.01, -0.01]).std()  # every 2-bar window has this std
    assert rolling_portfolio_vol_rmse(rewards, target_vol=window_vol, window=2, bars_per_year=1) == pytest.approx(0.0)


def test_rolling_portfolio_vol_rmse_measures_distance_from_target():
    rewards = pd.Series([0.01, -0.01] * 50)
    window_vol = pd.Series([0.01, -0.01]).std()
    rmse = rolling_portfolio_vol_rmse(rewards, target_vol=window_vol + 0.003, window=2, bars_per_year=1)
    assert rmse == pytest.approx(0.003)


def test_turnover_sums_absolute_position_changes():
    assert turnover(pd.Series([0.0, 0.5, -0.5, -0.5])) == pytest.approx(1.5)


def test_beta_alpha_of_scaled_market_is_scale_with_no_alpha():
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, 500))
    beta, alpha = beta_alpha(0.6 * market, market, bars_per_year=8760)
    assert beta == pytest.approx(0.6)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_beta_alpha_detects_constant_edge_on_top_of_market():
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, 500))
    beta, alpha = beta_alpha(0.5 * market + 1e-5, market, bars_per_year=8760)
    assert beta == pytest.approx(0.5)
    assert alpha == pytest.approx(1e-5 * 8760)
