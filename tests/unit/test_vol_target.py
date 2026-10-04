import math

import numpy as np
import pandas as pd
import pytest
import torch

from control.vol_target import VolTargetLaw


def _state(log_return=0.0, vol_forecast=0.5, trend=None):
    row = {"log_return": log_return, "vol_forecast": vol_forecast}
    if trend is not None:
        row["trend"] = trend
    return pd.Series(row)


def test_unit_kp_is_textbook_vol_targeting():
    law = VolTargetLaw(target_vol=0.25)
    ctx = law.begin_step(_state(vol_forecast=0.5))
    assert law.commit(ctx, kp=1.0, ki=0.0, kd=0.0) == pytest.approx(0.5)


def test_exposure_is_capped_at_max_position():
    law = VolTargetLaw(target_vol=0.5, max_position=1.5)
    ctx = law.begin_step(_state(vol_forecast=0.1))  # uncapped would be 5x
    assert law.commit(ctx, 1.0, 0.0, 0.0) == pytest.approx(1.5)


def test_kp_above_one_derisks_harder_in_high_vol():
    law = VolTargetLaw(target_vol=0.25)
    ctx = law.begin_step(_state(vol_forecast=1.0))
    base = law.position(ctx, 1.0, 0.0, 0.0)
    aggressive = law.position(ctx, 1.5, 0.0, 0.0)
    assert aggressive < base


def test_position_is_differentiable_in_tensor_gains():
    law = VolTargetLaw(target_vol=0.25)
    ctx = law.begin_step(_state(vol_forecast=0.5))
    kp = torch.tensor(1.0, requires_grad=True)
    pos = law.position(ctx, kp, 0.0, 0.0)
    pos.backward()
    # d/dkp exp(kp * log(0.5)) = log(0.5) * 0.5
    assert kp.grad.item() == pytest.approx(math.log(0.5) * 0.5)


def test_tensor_and_float_paths_agree():
    law = VolTargetLaw(target_vol=0.25)
    ctx = law.begin_step(_state(vol_forecast=0.4))
    as_float = law.position(ctx, 1.2, 0.0, 0.3)
    as_tensor = law.position(ctx, torch.tensor(1.2), torch.tensor(0.0), torch.tensor(0.3))
    assert float(as_tensor) == pytest.approx(as_float, rel=1e-6)


def test_integral_stays_off_until_portfolio_vol_window_is_full():
    law = VolTargetLaw(target_vol=0.25, port_vol_window=5)
    rng = np.random.default_rng(0)
    for _ in range(4):
        ctx = law.begin_step(_state(log_return=rng.normal(0, 0.01)))
        assert ctx.feedback_error == 0.0
        assert ctx.errors.e_int == 0.0
        law.commit(ctx, 1.0, 0.01, 0.0)
    # fifth booked return fills the window
    ctx = law.begin_step(_state(log_return=rng.normal(0, 0.01)))
    assert ctx.feedback_error != 0.0


def test_portfolio_vol_measures_own_returns():
    law = VolTargetLaw(target_vol=0.25, port_vol_window=3, bars_per_year=1.0)
    returns = [0.0, 0.1, -0.1, 0.1]  # the first is earned while flat
    for r in returns:
        ctx = law.begin_step(_state(log_return=r, vol_forecast=0.25))
        law.commit(ctx, 1.0, 0.0, 0.0)  # position 1.0 after the first bar
    expected = np.std([math.expm1(r) for r in returns[1:]], ddof=1)
    assert law.portfolio_vol() == pytest.approx(expected)


def test_integral_raises_exposure_when_portfolio_vol_runs_below_target():
    """vol_forecast overstates vol 2x, so feedforward under-sizes and realized
    portfolio vol stays below target. The integral term must push exposure up."""
    rng = np.random.default_rng(0)
    true_vol, target = 0.4, 0.2
    law = VolTargetLaw(target_vol=target, port_vol_window=48, max_position=3.0)
    positions = []
    for _ in range(2000):
        r = rng.normal(0, true_vol / np.sqrt(8760))
        ctx = law.begin_step(_state(log_return=r, vol_forecast=2 * true_vol))
        positions.append(law.commit(ctx, 1.0, 0.01, 0.0))
    feedforward = target / (2 * true_vol)
    assert np.mean(positions[-200:]) > 1.5 * feedforward


def test_integral_does_not_wind_up_while_exposure_is_capped():
    law = VolTargetLaw(target_vol=0.5, port_vol_window=10, max_position=1.0)
    rng = np.random.default_rng(1)
    for _ in range(300):
        # tiny returns: portfolio vol far below target, but exposure already capped
        ctx = law.begin_step(_state(log_return=rng.normal(0, 1e-4), vol_forecast=0.1))
        law.commit(ctx, 1.0, 0.01, 0.0)
    assert ctx.errors.e_int == pytest.approx(0.0)


def test_integral_is_clamped():
    law = VolTargetLaw(target_vol=0.01, port_vol_window=10, integral_limit=5.0)
    rng = np.random.default_rng(2)
    for _ in range(300):
        # portfolio vol far above target, every step adds a large negative error
        ctx = law.begin_step(_state(log_return=rng.normal(0, 0.02), vol_forecast=0.01))
        law.commit(ctx, 1.0, 0.0, 0.0)
    assert ctx.errors.e_int == pytest.approx(-5.0)


def test_reset_clears_state():
    law = VolTargetLaw(target_vol=0.25, port_vol_window=2)
    for r in [0.01, 0.02, -0.01, 0.03]:
        law.commit(law.begin_step(_state(log_return=r)), 1.0, 0.01, 0.0)
    law.reset()
    ctx = law.begin_step(_state(log_return=0.05))
    assert ctx.errors.e_int == 0.0
    assert ctx.errors.e_d == 0.0
    assert law.portfolio_vol() is None


def test_derivative_is_change_of_proportional_error():
    law = VolTargetLaw(target_vol=0.25)
    first = law.begin_step(_state(vol_forecast=0.5))
    law.commit(first, 1.0, 0.0, 0.0)
    second = law.begin_step(_state(vol_forecast=0.25))
    assert first.errors.e_d == 0.0
    assert second.errors.e_d == pytest.approx(math.log(1.0) - math.log(0.5))


@pytest.mark.parametrize("mode,trend,expected", [
    ("long_only", None, 1.0),
    ("trend_long_flat", 0.05, 1.0),
    ("trend_long_flat", -0.05, 0.0),
    ("trend_long_short", -0.05, -1.0),
])
def test_direction_modes(mode, trend, expected):
    law = VolTargetLaw(target_vol=0.5, direction_mode=mode)
    ctx = law.begin_step(_state(vol_forecast=0.5, trend=trend))
    assert law.commit(ctx, 1.0, 0.0, 0.0) == pytest.approx(expected)


def test_trend_modes_require_trend_column():
    assert "trend" in VolTargetLaw(target_vol=0.5, direction_mode="trend_long_flat").required_fields
    assert "trend" not in VolTargetLaw(target_vol=0.5).required_fields


@pytest.mark.parametrize("kwargs,match", [
    ({"target_vol": 0.0}, "target_vol must be positive"),
    ({"target_vol": 0.5, "direction_mode": "sideways"}, "unknown direction_mode"),
    ({"target_vol": 0.5, "port_vol_window": 1}, "port_vol_window"),
])
def test_invalid_config_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VolTargetLaw(**kwargs)



def test_direction_is_held_between_reads():
    law = VolTargetLaw(target_vol=0.5, direction_mode="trend_long_flat",
                       trend_column="mom", direction_hold=3)
    signals = [0.1, -0.1, -0.1, -0.1, 0.1, 0.1]
    directions = []
    for sig in signals:
        ctx = law.begin_step(pd.Series({"log_return": 0.0, "vol_forecast": 0.5, "mom": sig}))
        law.commit(ctx, 1.0, 0.0, 0.0)
        directions.append(ctx.direction)
    # read on bars 0 and 3 only
    assert directions == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_reset_forces_a_fresh_direction_read():
    law = VolTargetLaw(target_vol=0.5, direction_mode="trend_long_flat",
                       trend_column="mom", direction_hold=100)
    law.begin_step(pd.Series({"log_return": 0.0, "vol_forecast": 0.5, "mom": 0.1}))
    law.reset()
    ctx = law.begin_step(pd.Series({"log_return": 0.0, "vol_forecast": 0.5, "mom": -0.1}))
    assert ctx.direction == 0.0


def test_trend_column_is_configurable():
    law = VolTargetLaw(target_vol=0.5, direction_mode="trend_long_flat", trend_column="mom_720h")
    assert "mom_720h" in law.required_fields
