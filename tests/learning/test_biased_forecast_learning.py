"""E2E test for learning: A2CPID learns to correct a biased vol forecast.

Not a regression test in the usual sense, because it depends on hyperparameters and
seed, and can legitimately fail if either drift.

Scenario: the vol_forecast column overstates true vol by 2x, so the static
PID (kp=1, no integral) holds half the exposure needed to hit the target.
The exact fix within the law is kp = log(target/vol) / log(target/(2*vol)),
which is 0.5 for target=0.25, vol=0.5. Trained on the QLIKE tracking loss,
A2CPID should move kp toward it and track the target better than the
static PID on fresh data from the same market.
"""

import numpy as np
import pytest
import torch

from backtest.engine import run_backtest
from backtest.metrics import annualized_vol
from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw
from environment.trading_env import TradingEnv
from models.pid_net import PIDNet
from tests.synthetic import make_market
from training.train_loop import TrainConfig, train_one_pass


pytestmark = pytest.mark.slow


TARGET_VOL = 0.25
TRUE_VOL = 0.5
N_BARS = 3000
SEED = 42


def _law() -> VolTargetLaw:
    return VolTargetLaw(target_vol=TARGET_VOL, max_position=2.0)


def _market(seed: int):
    return make_market(n=N_BARS, vol=TRUE_VOL, seed=seed, vol_forecast=2 * TRUE_VOL)


def test_a2c_pid_learns_to_correct_biased_vol_forecast():
    torch.manual_seed(SEED)
    a2c = A2CPID(
        static_pid=StaticPID(law=_law(), kp=1.0),
        pid_net=PIDNet(gain_bounds=(0.75, 1e-6, 1e-6)),
    )
    train_env = TradingEnv(_market(SEED), fee_rate=0.0, max_position=2.0)
    reports = train_one_pass(
        train_env, a2c,
        TrainConfig(lr=1e-2, w_return=0.0, w_track=1.0, w_gain=0.0, epochs=2),
    )
    learned_kp = np.mean([r.kp for r in reports[-500:]])

    test_data = _market(SEED + 1)
    static_vol = annualized_vol(
        run_backtest(TradingEnv(test_data, fee_rate=0.0, max_position=2.0), StaticPID(law=_law(), kp=1.0)).rewards,
        bars_per_year=8760,
    )
    a2c_vol = annualized_vol(
        run_backtest(TradingEnv(test_data, fee_rate=0.0, max_position=2.0), a2c).rewards,
        bars_per_year=8760,
    )

    print(f"\nlearned kp (last 500 bars): {learned_kp:.3f}, ideal 0.5")
    print(f"realized vol: static={static_vol:.3f} a2c={a2c_vol:.3f} target={TARGET_VOL}")

    assert learned_kp < 0.8, f"kp did not move toward 0.5: {learned_kp:.3f}"
    assert abs(a2c_vol - TARGET_VOL) < abs(static_vol - TARGET_VOL)
