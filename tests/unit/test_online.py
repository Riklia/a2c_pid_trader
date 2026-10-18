import copy

import numpy as np
import pytest
import torch

from backtest.engine import run_backtest
from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw
from environment.trading_env import TradingEnv
from models.pid_net import PIDNet
from tests.synthetic import make_market
from training.online import run_online
from training.train_loop import TrainConfig


def _controller(seed=0, bounds=(0.5, 0.002, 5.0)):
    torch.manual_seed(seed)
    law = VolTargetLaw(target_vol=0.25, port_vol_window=24, max_position=2.0)
    return A2CPID(static_pid=StaticPID(law=law, kp=1.0), pid_net=PIDNet(gain_bounds=bounds))


def test_without_learning_online_equals_frozen_backtest():
    data = make_market(n=300, seed=1)
    frozen = _controller()
    online = copy.deepcopy(frozen)

    expected = run_backtest(TradingEnv(data, fee_rate=4e-4, max_position=2.0), frozen)
    actual = run_online(TradingEnv(data, fee_rate=4e-4, max_position=2.0), online, TrainConfig(lr=0.0))

    np.testing.assert_allclose(actual.actions.to_numpy(), expected.actions.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(actual.equity.to_numpy(), expected.equity.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(actual.rewards.to_numpy(), expected.rewards.to_numpy(), atol=1e-12)
    assert list(actual.rewards.index) == list(expected.rewards.index)


def test_online_never_uses_future_data():
    data = make_market(n=300, seed=2)
    cutoff = 180
    config = TrainConfig(lr=1e-2, update_every=1)

    full = run_online(TradingEnv(data, fee_rate=0.0, max_position=2.0), _controller(seed=3), config)
    cut = run_online(TradingEnv(data.iloc[:cutoff], fee_rate=0.0, max_position=2.0), _controller(seed=3), config)

    np.testing.assert_allclose(full.actions.iloc[: cutoff - 1].to_numpy(), cut.actions.to_numpy(), atol=1e-9)


def test_online_forces_a_single_pass():
    data = make_market(n=50, seed=4)
    result = run_online(TradingEnv(data, fee_rate=0.0, max_position=2.0), _controller(), TrainConfig(epochs=3))
    assert len(result.actions) == len(data) - 1


@pytest.mark.slow
def test_online_adapts_to_a_forecast_bias_that_appears_after_training():
    """Trained on an unbiased market, then the vol forecast starts overstating
    vol 2x. Frozen, the controller keeps holding half the needed exposure;
    online, it learns to lift kp's correction and realized vol moves back to target."""
    config = TrainConfig(lr=1e-2, w_return=0.0, w_track=1.0, w_gain=0.0)
    controller = _controller(bounds=(0.75, 1e-6, 1e-6))
    train = make_market(n=2000, vol=0.5, seed=5)
    run_online(TradingEnv(train, fee_rate=0.0, max_position=2.0), controller, config)

    shifted = make_market(n=3000, vol=0.5, seed=6, vol_forecast=1.0)
    frozen = run_backtest(TradingEnv(shifted, fee_rate=0.0, max_position=2.0), copy.deepcopy(controller))
    online = run_online(TradingEnv(shifted, fee_rate=0.0, max_position=2.0), controller, config)

    late = slice(-1000, None)
    vol = lambda r: r.rewards.iloc[late].std() * np.sqrt(8760)
    assert abs(vol(online) - 0.25) < 0.5 * abs(vol(frozen) - 0.25)


def test_online_accepts_and_advances_carried_optimizer_state():
    data = make_market(n=100, seed=7)
    ctrl = _controller(seed=7)
    cfg = TrainConfig(lr=1e-2, update_every=24)
    opt = torch.optim.Adam(ctrl.pid_net.parameters(), lr=cfg.lr)

    run_online(TradingEnv(data.iloc[:50], fee_rate=0.0, max_position=2.0), ctrl, cfg, optimizer=opt)
    first_param = next(iter(ctrl.pid_net.parameters()))
    steps_after_train = int(opt.state[first_param]["step"].item())
    assert steps_after_train > 0

    run_online(TradingEnv(data.iloc[50:], fee_rate=0.0, max_position=2.0), ctrl, cfg, optimizer=opt)
    steps_after_online = int(opt.state[first_param]["step"].item())
    assert steps_after_online > steps_after_train
