import numpy as np
import pandas as pd
import pytest
import torch

from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw
from environment.trading_env import TradingEnv
from models.pid_net import PIDNet
from tests.synthetic import make_market
from training.train_loop import TrainConfig, train_interleaved, train_one_pass


def _fresh_controller(seed=0, port_vol_window=24):
    torch.manual_seed(seed)
    law = VolTargetLaw(target_vol=0.25, port_vol_window=port_vol_window)
    return A2CPID(
        static_pid=StaticPID(law=law, kp=1.0, ki=0.002),
        pid_net=PIDNet(gain_bounds=(0.5, 0.002, 5.0)),
    )


def test_train_one_pass_runs_to_end():
    env = TradingEnv(make_market(n=60), fee_rate=0.0)
    reports = train_one_pass(env, _fresh_controller(), TrainConfig())
    assert env.done
    assert len(reports) == len(env.data) - 1


def test_epochs_repeat_the_pass():
    env = TradingEnv(make_market(n=30), fee_rate=0.0)
    reports = train_one_pass(env, _fresh_controller(), TrainConfig(epochs=3))
    assert len(reports) == 3 * (len(env.data) - 1)


def test_reports_carry_step_metadata():
    data = make_market(n=10)
    env = TradingEnv(data, fee_rate=0.0)
    reports = train_one_pass(env, _fresh_controller(), TrainConfig())

    first = reports[0]
    assert first.ts == data.index[1]
    assert 0.0 <= first.action <= 1.0
    assert all(np.isfinite([r.utility, r.loss, r.kp, r.ki, r.kd]).all() for r in reports)


def test_pid_net_changes_after_training():
    env = TradingEnv(make_market(n=200), fee_rate=0.0004)
    ctrl = _fresh_controller()
    before = [p.detach().clone() for p in ctrl.pid_net.parameters()]

    train_one_pass(env, ctrl, TrainConfig(lr=1e-2))

    assert any(not torch.allclose(b, a) for b, a in zip(before, ctrl.pid_net.parameters()))


def test_on_step_callback_receives_reports_in_order():
    env = TradingEnv(make_market(n=8), fee_rate=0.0)
    seen = []
    train_one_pass(env, _fresh_controller(), TrainConfig(), on_step=seen.append)

    assert [r.step for r in seen] == list(range(1, len(env.data)))


def test_train_loop_has_no_look_ahead_on_actions():
    """Actions taken before a cutoff must not depend on data after it.

    update_every=1 so both runs take identical optimizer steps up to the cutoff.
    """
    full_data = make_market(n=300, seed=42)
    cutoff_idx = 200
    config = TrainConfig(lr=5e-3, update_every=1)

    env_full = TradingEnv(full_data, fee_rate=0.0)
    train_one_pass(env_full, _fresh_controller(seed=42), config)

    env_trunc = TradingEnv(full_data.iloc[:cutoff_idx], fee_rate=0.0)
    train_one_pass(env_trunc, _fresh_controller(seed=42), config)

    actions_full = env_full.position_history[:cutoff_idx]
    actions_trunc = env_trunc.position_history[:cutoff_idx]
    for i, (a, b) in enumerate(zip(actions_full, actions_trunc)):
        assert abs(a - b) < 1e-9, f"look-ahead detected at step {i}: {a} vs {b}"


def _shared_streams(datasets, seed=0):
    torch.manual_seed(seed)
    net = PIDNet(gain_bounds=(0.5, 0.002, 5.0))
    streams = {}
    for name, data in datasets.items():
        law = VolTargetLaw(target_vol=0.25, port_vol_window=24)
        streams[name] = (TradingEnv(data, fee_rate=0.0), A2CPID(static_pid=StaticPID(law=law), pid_net=net))
    return streams, net


def test_interleaved_training_steps_every_asset_through_every_bar():
    streams, _ = _shared_streams({"a": make_market(n=50, seed=1), "b": make_market(n=80, vol=0.9, seed=2)})
    reports = train_interleaved(streams, TrainConfig())
    assert all(env.done for env, _ in streams.values())
    assert sum(r.asset == "a" for r in reports) == 49
    assert sum(r.asset == "b" for r in reports) == 79


def test_interleaved_training_follows_calendar_order():
    late = make_market(n=40, seed=3)
    late.index = late.index + pd.Timedelta(hours=20)  # starts 20 bars after the other asset
    streams, _ = _shared_streams({"early": make_market(n=40, seed=4), "late": late})
    reports = train_interleaved(streams, TrainConfig())
    timestamps = [r.ts for r in reports]
    assert timestamps == sorted(timestamps)


def test_interleaved_training_requires_a_shared_network():
    a = A2CPID(static_pid=StaticPID(law=VolTargetLaw(target_vol=0.25)), pid_net=PIDNet())
    b = A2CPID(static_pid=StaticPID(law=VolTargetLaw(target_vol=0.25)), pid_net=PIDNet())
    streams = {"a": (TradingEnv(make_market(n=10), fee_rate=0.0), a),
               "b": (TradingEnv(make_market(n=10), fee_rate=0.0), b)}
    with pytest.raises(ValueError, match="share one pid_net"):
        train_interleaved(streams, TrainConfig())


def test_interleaved_training_has_no_cross_asset_look_ahead():
    """Asset a's actions before a cutoff must not depend on asset b's data after it."""
    a = make_market(n=200, seed=5)
    b = make_market(n=200, vol=0.8, seed=6)
    cutoff = 120
    config = TrainConfig(lr=5e-3, update_every=1)

    full, _ = _shared_streams({"a": a, "b": b}, seed=7)
    train_interleaved(full, config)
    truncated, _ = _shared_streams({"a": a.iloc[:cutoff], "b": b.iloc[:cutoff]}, seed=7)
    train_interleaved(truncated, config)

    actions_full = full["a"][0].position_history[:cutoff]
    actions_trunc = truncated["a"][0].position_history[:cutoff]
    assert np.allclose(actions_full, actions_trunc, atol=1e-9)
