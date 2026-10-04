import math

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest, BacktestResult
from control.static_pid import StaticPID
from control.baselines import BuyAndHold
from control.vol_target import VolTargetLaw
from environment.trading_env import TradingEnv


def _make_data(log_returns, vol_forecast=0.5):
    n = len(log_returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "close": [100.0] * n,
        "log_return": log_returns,
        "funding": [0.0] * n,
        "is_funding_settlement": [False] * n,
        "vol_forecast": [vol_forecast] * n,
    }, index=idx)


def test_backtest_runs_to_end():
    data = _make_data([np.nan, 0.01, 0.01, 0.01])
    env = TradingEnv(data, fee_rate=0.0)

    result = run_backtest(env, BuyAndHold())

    assert env.done
    assert len(result.rewards) == len(data) - 1
    assert len(result.equity) == len(data)


def test_backtest_result_types():
    data = _make_data([np.nan, 0.01, 0.01])
    result = run_backtest(TradingEnv(data), BuyAndHold())

    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity.index, pd.DatetimeIndex)
    assert isinstance(result.positions.index, pd.DatetimeIndex)


def test_flat_controller_leaves_equity_unchanged():
    data = _make_data([np.nan, 0.5, -0.3, 0.2])
    result = run_backtest(TradingEnv(data, fee_rate=0.0), BuyAndHold(position=0.0))

    assert (result.equity == 1.0).all()
    assert (result.positions == 0.0).all()


def test_vol_targeting_position_is_applied_to_next_bar():
    data = _make_data(log_returns=[np.nan, 0.1, 0.0], vol_forecast=0.5)
    controller = StaticPID(law=VolTargetLaw(target_vol=0.25))

    result = run_backtest(TradingEnv(data, fee_rate=0.0), controller)

    # position 0.25 / 0.5 = 0.5 chosen at bar 0, earns bar 1's simple return
    assert result.actions.iloc[0] == pytest.approx(0.5)
    assert result.equity.iloc[1] == pytest.approx(1.0 + 0.5 * math.expm1(0.1))


def test_backtest_resets_controller_state():
    data = _make_data([np.nan, 0.05, -0.04, 0.03, -0.02])
    controller = StaticPID(law=VolTargetLaw(target_vol=0.25, port_vol_window=2), ki=0.5)

    first = run_backtest(TradingEnv(data, fee_rate=0.0), controller)
    second = run_backtest(TradingEnv(data, fee_rate=0.0), controller)

    pd.testing.assert_series_equal(first.actions, second.actions)
