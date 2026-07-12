"""E2E test for learning: A2CPID learns to adapt to a regime shift better than
a fixed-gain PID.

Not a regression test in the usual sense, because it depends on hyperparameters and
seed, and can legitimately fail if either drift.

Scenario: synthetic data with two volatility regimes. Optimal position sizing
is nonlinear (|position| = target_vol / realized_vol, clipped to [-1, 1]),
so a fixed linear PID cannot be optimal in both regimes at once. A2CPID
should adapt online and score a lower portfolio-vol tracking RMSE on the
post-shift regime.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from backtest.engine import run_backtest
from backtest.metrics import portfolio_vol_tracking_rmse
from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from environment.trading_env import TradingEnv
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet
from data.features import compute_pid_errors
from training.train_loop import TrainConfig, train_one_pass


pytestmark = pytest.mark.slow


TARGET_VOL = 0.6
VOL_LOW = 0.3
VOL_HIGH = 1.2
N_BARS = 3000
REGIME_SHIFT_AT = 1500
SEED = 42


def _make_regime_shift_data(n: int, shift_at: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    true_vol = np.where(np.arange(n) < shift_at, VOL_LOW, VOL_HIGH)
    hourly_std = true_vol / np.sqrt(8760)
    log_returns = rng.normal(0, hourly_std, size=n)
    log_returns[0] = np.nan

    realized_vol = pd.Series(log_returns).rolling(24, min_periods=24).std().to_numpy() * np.sqrt(8760)
    realized_vol_series = pd.Series(realized_vol, index=idx)
    errors = compute_pid_errors(realized_vol_series, target_vol=TARGET_VOL,
                                integral_mode="rolling", integral_window="720h")
    e_p = errors["e_p"].to_numpy()
    e_i = errors["e_i"].fillna(0.0).to_numpy()  # first rows have NaN before rolling window fills
    e_d = errors["e_d"].fillna(0.0).to_numpy()
    close = 100 * np.exp(np.cumsum(np.nan_to_num(log_returns)))

    df = pd.DataFrame({
        "close": close,
        "log_return": log_returns,
        "funding": np.zeros(n),
        "is_funding_settlement": np.zeros(n, dtype=bool),
        "realized_vol": realized_vol,
        "e_p": e_p,
        "e_i": e_i,
        "e_d": e_d,
    }, index=idx)
    return df.dropna()


def _fresh_a2c(seed: int) -> A2CPID:
    torch.manual_seed(seed)
    return A2CPID(
        static_pid=StaticPID(kp=1.0),
        pid_net=PIDNet(),
        actor=Actor(),
        critic=Critic(),
    )


def _post_shift_rmse(positions: pd.Series, data: pd.DataFrame, shift_at: int) -> float:
    """Portfolio-vol tracking RMSE, measured only on the post-shift regime."""
    aligned_positions = positions.iloc[shift_at:]
    aligned_vol = data["realized_vol"].iloc[shift_at:]
    return portfolio_vol_tracking_rmse(aligned_positions, aligned_vol, TARGET_VOL)

def test_a2c_pid_beats_static_on_post_shift_regime():
    data = _make_regime_shift_data(N_BARS, REGIME_SHIFT_AT, SEED)

    static_env = TradingEnv(data, fee_rate=0.0)
    static_ctrl = StaticPID(kp=1.0, ki=0.1)
    static_result = run_backtest(static_env, static_ctrl)

    a2c_env = TradingEnv(data, fee_rate=0.0)
    a2c_ctrl = _fresh_a2c(SEED)
    train_one_pass(
        a2c_env, a2c_ctrl,
        TrainConfig(lr=5e-3, w_error=1.0, w_action=0.05, w_entropy=0.001),
        target_vol=TARGET_VOL,
    )
    a2c_positions = pd.Series(a2c_env.position_history, index=a2c_env.data.index[: len(a2c_env.position_history)])

    static_rmse = _post_shift_rmse(static_result.positions, data, REGIME_SHIFT_AT)
    a2c_rmse = _post_shift_rmse(a2c_positions, data, REGIME_SHIFT_AT)
    print(f"\nstatic PID RMSE (post-shift): {static_rmse:.4f}")
    print(f"A2C-PID RMSE   (post-shift): {a2c_rmse:.4f}")
    print(f"relative improvement:        {(static_rmse - a2c_rmse) / static_rmse:.1%}")

    assert a2c_rmse < static_rmse, (
        f"A2CPID did not improve over static baseline: "
        f"a2c_rmse={a2c_rmse:.4f} vs static_rmse={static_rmse:.4f}"
    )
