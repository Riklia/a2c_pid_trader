import math

import pandas as pd
import pytest

from control.controller_base import Controller
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw


def _state(log_return=0.0, vol_forecast=0.5):
    return pd.Series({"log_return": log_return, "vol_forecast": vol_forecast})


def test_default_gains_give_plain_vol_targeting():
    pid = StaticPID(law=VolTargetLaw(target_vol=0.2))
    assert pid.act(_state(vol_forecast=0.8)) == pytest.approx(0.25)


def test_gains_change_the_position():
    state = _state(vol_forecast=0.8)
    soft = StaticPID(law=VolTargetLaw(target_vol=0.2), kp=0.5).act(state)
    hard = StaticPID(law=VolTargetLaw(target_vol=0.2), kp=1.5).act(state)
    assert soft == pytest.approx(math.exp(0.5 * math.log(0.25)))
    assert hard < 0.25 < soft


def test_position_never_exceeds_cap():
    pid = StaticPID(law=VolTargetLaw(target_vol=0.5, max_position=1.0))
    assert pid.act(_state(vol_forecast=0.05)) == 1.0


def test_missing_state_field_raises_with_context():
    pid = StaticPID(law=VolTargetLaw(target_vol=0.5))
    with pytest.raises(ValueError, match="StaticPID.act.*missing required fields"):
        pid.act(pd.Series({"log_return": 0.0}))


def test_reset_restarts_the_law():
    pid = StaticPID(law=VolTargetLaw(target_vol=0.5, port_vol_window=2), ki=0.1)
    for r in [0.01, -0.02, 0.03, -0.01]:
        pid.act(_state(log_return=r))
    pid.reset()
    assert pid.law.portfolio_vol() is None


def test_satisfies_controller_protocol():
    assert isinstance(StaticPID(law=VolTargetLaw(target_vol=0.5)), Controller)
