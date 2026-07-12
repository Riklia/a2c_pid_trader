import numpy as np
import pandas as pd
import pytest

from environment.trading_env import TradingEnv


def _make_data(log_returns, funding=None, is_settlement=None):
    n = len(log_returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "close": [100.0] * n,
        "log_return": log_returns,
        "funding": funding or [0.0] * n,
        "is_funding_settlement": is_settlement or [False] * n,
    }, index=idx)


def test_missing_columns_raises():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="1h"))
    with pytest.raises(ValueError, match="missing required columns"):
        TradingEnv(df)


def test_position_earns_next_bar_return_not_its_own():
    data = _make_data(log_returns=[np.nan, 1.0, 0.0])
    env = TradingEnv(data, fee_rate=0.0)

    _, reward, _ = env.step(target_position=1.0)

    # position=1, next-bar log_return=1.0 -> equity += 1.0 * equity -> equity=2.0
    assert env.equity == pytest.approx(2.0)
    assert reward == pytest.approx(np.log(2.0))


def test_flat_position_is_unaffected_by_price_moves():
    data = _make_data(log_returns=[np.nan, 0.5, -0.3])
    env = TradingEnv(data, fee_rate=0.0)

    env.step(target_position=0.0)
    assert env.equity == pytest.approx(1.0)


def test_fee_charged_on_turnover_when_opening_position():
    data = _make_data(log_returns=[np.nan, 0.0, 0.0])
    env = TradingEnv(data, fee_rate=0.001, initial_equity=1.0)

    env.step(target_position=1.0)  # turnover = 1.0 (from 0 to 1)

    assert env.equity == pytest.approx(1.0 * (1 - 0.001))


def test_fee_charged_only_on_position_change():
    data = _make_data(log_returns=[np.nan, 0.0, 0.0, 0.0])
    env = TradingEnv(data, fee_rate=0.001)

    env.step(target_position=1.0)
    equity_after_first = env.equity
    env.step(target_position=1.0)  # no change -> no fee

    assert env.equity == pytest.approx(equity_after_first)


def test_funding_charged_only_on_settlement_bar():
    data = _make_data(
        log_returns=[np.nan, 0.0, 0.0],
        funding=[0.0, 0.01, 0.01],
        is_settlement=[False, False, True],
    )
    env = TradingEnv(data, fee_rate=0.0)

    # moves to row1: funding due but not a settlement bar
    env.step(target_position=1.0)
    equity_after_first = env.equity
    assert equity_after_first == pytest.approx(1.0)

    # moves to row2: settlement bar, funding charged
    env.step(target_position=1.0)
    assert env.equity == pytest.approx(equity_after_first * (1 - 0.01))


def test_short_position_gains_on_price_drop():
    data = _make_data(log_returns=[np.nan, -0.5, 0.0])
    env = TradingEnv(data, fee_rate=0.0)

    env.step(target_position=-1.0)

    # position=-1, log_return=-0.5 -> equity += (-1)*(-0.5)*equity -> equity=1.5
    assert env.equity == pytest.approx(1.5)


def test_target_position_is_clipped_to_valid_range():
    data = _make_data(log_returns=[np.nan, 0.0, 0.0])
    env = TradingEnv(data, fee_rate=0.0)

    env.step(target_position=5.0)
    assert env.position == 1.0


def test_step_after_done_raises():
    data = _make_data(log_returns=[np.nan, 0.0])
    env = TradingEnv(data, fee_rate=0.0)

    env.step(target_position=1.0)
    assert env.done
    with pytest.raises(RuntimeError, match="already ended"):
        env.step(target_position=0.5)


def test_reset_restores_initial_state():
    data = _make_data(log_returns=[np.nan, 0.5, -0.3])
    env = TradingEnv(data, fee_rate=0.001)

    env.step(target_position=1.0)
    env.reset()

    assert env.equity == 1.0
    assert env.position == 0.0
    assert env.equity_curve == [1.0]