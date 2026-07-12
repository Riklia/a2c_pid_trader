import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest, BacktestResult
from control.static_pid import StaticPID
from environment.trading_env import TradingEnv


def _make_data(log_returns, e_p=None, e_i=None, e_d=None):
    n = len(log_returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "close": [100.0] * n,
        "log_return": log_returns,
        "funding": [0.0] * n,
        "is_funding_settlement": [False] * n,
        "e_p": e_p if e_p is not None else [0.0] * n,
        "e_i": e_i if e_i is not None else [0.0] * n,
        "e_d": e_d if e_d is not None else [0.0] * n,
    }, index=idx)


def test_backtest_runs_to_end():
    data = _make_data([np.nan, 0.01, 0.01, 0.01])
    env = TradingEnv(data, fee_rate=0.0)
    controller = StaticPID(kp=1.0)  # e_p=0 -> always position=0

    result = run_backtest(env, controller)

    assert env.done
    assert len(result.rewards) == len(data) - 1
    assert len(result.equity) == len(data)


def test_backtest_result_types():
    data = _make_data([np.nan, 0.01, 0.01])
    env = TradingEnv(data)
    controller = StaticPID(kp=1.0)

    result = run_backtest(env, controller)

    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity.index, pd.DatetimeIndex)
    assert isinstance(result.positions.index, pd.DatetimeIndex)


def test_flat_controller_leaves_equity_unchanged():
    """e_p is always zero -> StaticPID always outputs 0 -> equity is flat."""
    data = _make_data([np.nan, 0.5, -0.3, 0.2])
    env = TradingEnv(data, fee_rate=0.0)
    controller = StaticPID(kp=1.0)

    result = run_backtest(env, controller)

    assert (result.equity == 1.0).all()
    assert (result.positions == 0.0).all()


def test_positive_error_makes_controller_go_long():
    data = _make_data(
        log_returns=[np.nan, 0.1, 0.0],
        e_p=[0.5, 0.5, 0.5],
    )
    env = TradingEnv(data, fee_rate=0.0)
    controller = StaticPID(kp=1.0)

    result = run_backtest(env, controller)

    # first action taken at step 0 based on e_p=0.5 -> position 0.5, applied against bar1's return 0.1
    # equity_after_step0 = 1.0 * (1 + 0.5 * 0.1) = 1.05
    assert result.equity.iloc[1] == pytest.approx(1.05)
    assert result.actions.iloc[0] == pytest.approx(0.5)