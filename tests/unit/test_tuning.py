import math

import numpy as np
import pandas as pd
import pytest

from control import tuning
from control.vol_target import VolTargetLaw


def test_kp_one_reproduces_vol_targeting():
    law = VolTargetLaw(target_vol=0.3)
    ctx = law.begin_step(pd.Series({"log_return": 0.0, "vol_forecast": 0.6}))
    assert law.position(ctx, tuning.kp_vol_targeting(), 0.0, 0.0) == pytest.approx(0.3 / 0.6)


def test_simc_default_is_inverse_window():
    assert tuning.ki_simc(168) == pytest.approx(1 / 168)


def test_simc_slower_closed_loop_lowers_gain():
    theta = tuning.sensor_dead_time(168)
    assert tuning.ki_simc(168, tau_c=3 * theta) == pytest.approx(1 / (4 * theta))


def test_simc_scales_with_plant_gain():
    assert tuning.ki_simc(100, plant_gain=2.0) == pytest.approx(tuning.ki_simc(100) / 2)


def test_integral_limit_caps_correction_factor():
    ki = 0.005
    limit = tuning.integral_limit_for(1.5, ki)
    assert math.exp(ki * limit) == pytest.approx(1.5)


@pytest.mark.parametrize("max_correction,ki", [(1.0, 0.01), (1.5, 0.0)])
def test_integral_limit_rejects_degenerate_inputs(max_correction, ki):
    with pytest.raises(ValueError):
        tuning.integral_limit_for(max_correction, ki)


def test_derived_integral_loop_settles_without_oscillation():
    """With the derived ki, a 2x biased vol forecast is corrected and the
    correction converges instead of oscillating."""
    window, target, true_vol = 48, 0.2, 0.4
    gains = tuning.derived_gains(window)
    law = VolTargetLaw(
        target_vol=target, port_vol_window=window, max_position=3.0,
        integral_limit=tuning.integral_limit_for(3.0, gains["ki"]),
    )
    rng = np.random.default_rng(0)
    positions = []
    for _ in range(4000):
        r = rng.normal(0, true_vol / np.sqrt(8760))
        ctx = law.begin_step(pd.Series({"log_return": r, "vol_forecast": 2 * true_vol}))
        positions.append(law.commit(ctx, **gains))
    ideal = target / true_vol
    late = np.array(positions[-1000:])
    assert late.mean() == pytest.approx(ideal, rel=0.2)
    assert late.std() < 0.25 * ideal
