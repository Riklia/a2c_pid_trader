import os

import numpy as np
import pytest

from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from data import loader, features
from environment.trading_env import TradingEnv
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet
from training.train_loop import TrainConfig, train_one_pass


pytestmark = pytest.mark.integration


TARGET_VOL = 0.6
FREQ = "1h"
VOL_WINDOW = "24h"
INTEGRAL_WINDOW = "720h"

# ~500 hourly bars starting from a known-populated window in the DB.
# Timestamp corresponds to 2023-11-14 22:13:20 UTC, safely inside the
# BTCUSDT perpetual coverage (min available timestamp in the dataset ~2019-09).
START_TS = 1700000000
END_TS = START_TS + 500 * 3600


@pytest.fixture(scope="module")
def engine():
    dsn = os.environ.get("DEV_DB_DSN")
    if not dsn:
        pytest.skip("DEV_DB_DSN not set, skipping integration test")
    return loader.get_engine(loader.DBConfig(dsn=dsn))


@pytest.fixture(scope="module")
def real_feature_table(engine):
    trades = loader.load_trades(engine, symbol="BTCUSDT", start_ts=START_TS, end_ts=END_TS)
    funding = loader.load_funding(engine, symbol="BTCUSDT", start_ts=START_TS, end_ts=END_TS)

    if trades.empty:
        pytest.skip(f"no trade data returned for the chosen window [{START_TS}, {END_TS}]")

    return features.build_feature_table(
        trades, funding,
        target_vol=TARGET_VOL, freq=FREQ,
        vol_window=VOL_WINDOW, integral_mode="rolling", integral_window=INTEGRAL_WINDOW,
    )


def _fresh_controller() -> A2CPID:
    return A2CPID(
        static_pid=StaticPID(kp=1.0),
        pid_net=PIDNet(),
        actor=Actor(),
        critic=Critic(),
    )


def test_feature_table_is_nonempty_and_shaped_correctly(real_feature_table):
    assert len(real_feature_table) > 50, "feature table too short to be useful"
    for col in ["close", "log_return", "funding", "is_funding_settlement",
                "realized_vol", "e_p", "e_i", "e_d"]:
        assert col in real_feature_table.columns
    assert real_feature_table.isna().sum().sum() == 0


def test_train_loop_completes_full_pass(real_feature_table):
    env = TradingEnv(real_feature_table, fee_rate=0.0004)
    ctrl = _fresh_controller()

    reports = train_one_pass(env, ctrl, TrainConfig(), target_vol=TARGET_VOL)

    assert env.done
    assert len(reports) == len(real_feature_table) - 1


def test_all_reports_carry_finite_numbers(real_feature_table):
    env = TradingEnv(real_feature_table, fee_rate=0.0004)
    ctrl = _fresh_controller()

    reports = train_one_pass(env, ctrl, TrainConfig(), target_vol=TARGET_VOL)

    for r in reports:
        assert np.isfinite(r.reward), f"non-finite reward at step {r.step}"
        assert np.isfinite(r.td), f"non-finite TD error at step {r.step}"
        assert np.isfinite(r.actor_loss), f"non-finite actor loss at step {r.step}"
        assert np.isfinite(r.critic_loss), f"non-finite critic loss at step {r.step}"
        assert np.isfinite(r.predicted_vol), f"non-finite predicted_vol at step {r.step}"
        assert -1.0 <= r.action <= 1.0, f"action out of bounds at step {r.step}: {r.action}"


def test_all_reports_reference_valid_bar_timestamps(real_feature_table):
    env = TradingEnv(real_feature_table, fee_rate=0.0004)
    ctrl = _fresh_controller()

    reports = train_one_pass(env, ctrl, TrainConfig(), target_vol=TARGET_VOL)

    valid_index = set(real_feature_table.index)
    for r in reports:
        assert r.ts in valid_index, f"step {r.step} references timestamp {r.ts} not in feature table"


def test_networks_actually_updated_after_full_pass(real_feature_table):
    env = TradingEnv(real_feature_table, fee_rate=0.0004)
    ctrl = _fresh_controller()

    def snapshot(module):
        return [p.detach().clone() for p in module.parameters()]

    before = {
        "actor": snapshot(ctrl.actor),
        "critic": snapshot(ctrl.critic),
        "pid_net": snapshot(ctrl.pid_net),
    }

    train_one_pass(env, ctrl, TrainConfig(lr=1e-3), target_vol=TARGET_VOL)

    import torch
    for name, module in [("actor", ctrl.actor), ("critic", ctrl.critic), ("pid_net", ctrl.pid_net)]:
        changed = any(
            not torch.allclose(before_p, after_p)
            for before_p, after_p in zip(before[name], module.parameters())
        )
        assert changed, f"{name} weights did not change after training"


def test_train_loop_shows_no_look_ahead_from_state_perspective(real_feature_table):
    env = TradingEnv(real_feature_table, fee_rate=0.0)
    ctrl = _fresh_controller()

    reports = train_one_pass(env, ctrl, TrainConfig(), target_vol=TARGET_VOL)

    # report[i].ts is the timestamp of the bar that produced the reward,
    # which should be real_feature_table.index[i + 1] since the very first
    # state (index 0) was seen but never became the "next" bar
    for i, r in enumerate(reports):
        expected_ts = real_feature_table.index[i + 1]
        assert r.ts == expected_ts, f"step {i}: expected ts {expected_ts}, got {r.ts}"
        assert r.actual_vol == pytest.approx(real_feature_table.iloc[i + 1]["realized_vol"])
