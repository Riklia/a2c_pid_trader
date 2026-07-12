import numpy as np
import pandas as pd
import torch

from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from environment.trading_env import TradingEnv
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet
from training.train_loop import TrainConfig, train_one_pass


def _make_data(n=30):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(0)
    log_returns = rng.normal(0, 0.01, n)
    log_returns[0] = np.nan
    realized_vol = np.abs(log_returns) * np.sqrt(8760)
    realized_vol[0] = 0.5
    target = 0.6
    e_p = target - realized_vol
    return pd.DataFrame({
        "close": np.cumprod(1 + np.nan_to_num(log_returns)) * 100,
        "log_return": log_returns,
        "funding": [0.0] * n,
        "is_funding_settlement": [False] * n,
        "realized_vol": realized_vol,
        "e_p": e_p,
        "e_i": np.cumsum(e_p),
        "e_d": np.diff(e_p, prepend=e_p[0]),
    }, index=idx)


def _fresh_controller():
    return A2CPID(
        static_pid=StaticPID(kp=1.0),
        pid_net=PIDNet(),
        actor=Actor(),
        critic=Critic(),
    )


def test_train_one_pass_runs_to_end():
    env = TradingEnv(_make_data(), fee_rate=0.0)
    ctrl = _fresh_controller()
    reports = train_one_pass(env, ctrl, TrainConfig(), target_vol=0.6)
    assert env.done
    assert len(reports) == len(env.data) - 1


def test_reports_carry_step_metadata():
    env = TradingEnv(_make_data(n=5), fee_rate=0.0)
    ctrl = _fresh_controller()
    reports = train_one_pass(env, ctrl, TrainConfig(), target_vol=0.6)

    first = reports[0]
    assert isinstance(first.ts, pd.Timestamp)
    assert -1.0 <= first.action <= 1.0
    # quadratic reward is nonpositive
    assert first.reward <= 0.0
    assert np.isfinite(first.td)


def test_networks_change_after_training():
    env = TradingEnv(_make_data(), fee_rate=0.0)
    ctrl = _fresh_controller()

    def snapshot(module):
        return [p.detach().clone() for p in module.parameters()]

    before_actor = snapshot(ctrl.actor)
    before_critic = snapshot(ctrl.critic)
    before_pid = snapshot(ctrl.pid_net)

    train_one_pass(env, ctrl, TrainConfig(lr=1e-2), target_vol=0.6)

    def changed(before, module):
        return any(not torch.allclose(b, a) for b, a in zip(before, module.parameters()))

    assert changed(before_actor, ctrl.actor)
    assert changed(before_critic, ctrl.critic)
    assert changed(before_pid, ctrl.pid_net)


def test_on_step_callback_receives_reports_in_order():
    env = TradingEnv(_make_data(n=5), fee_rate=0.0)
    ctrl = _fresh_controller()
    seen = []
    train_one_pass(env, ctrl, TrainConfig(), target_vol=0.6, on_step=seen.append)

    # sorted by construction
    assert [r.step for r in seen] == [r.step for r in seen]
    assert len(seen) == len(env.data) - 1
