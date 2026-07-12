import pandas as pd
import pytest

from control.static_pid import StaticPID


def _state(e_p=0.0, e_i=0.0, e_d=0.0) -> pd.Series:
    return pd.Series({"e_p": e_p, "e_i": e_i, "e_d": e_d})


def test_zero_error_gives_zero_position():
    pid = StaticPID(kp=1.0, ki=0.5, kd=0.2)
    assert pid.act(_state()) == 0.0


def test_positive_error_gives_positive_position():
    """realized_vol below target -> e_p > 0 -> controller increases exposure."""
    pid = StaticPID(kp=1.0, ki=0.0, kd=0.0)
    assert pid.act(_state(e_p=0.5)) == pytest.approx(0.5)


def test_negative_error_gives_negative_position():
    """realized_vol above target -> e_p < 0 -> controller reduces exposure."""
    pid = StaticPID(kp=1.0, ki=0.0, kd=0.0)
    assert pid.act(_state(e_p=-0.3)) == pytest.approx(-0.3)


def test_output_is_clipped_to_bounds():
    pid = StaticPID(kp=10.0, ki=0.0, kd=0.0)  # huge gain forces saturation
    assert pid.act(_state(e_p=0.5)) == 1.0
    assert pid.act(_state(e_p=-0.5)) == -1.0


def test_custom_bounds_are_respected():
    pid = StaticPID(kp=1.0, position_bounds=(0.0, 1.0))
    assert pid.act(_state(e_p=-0.5)) == 0.0
    assert pid.act(_state(e_p=0.5)) == pytest.approx(0.5)


def test_all_three_terms_combine_linearly():
    pid = StaticPID(kp=1.0, ki=0.5, kd=0.25)
    # 1.0 * 0.4 + 0.5 * 0.6 + 0.25 * (-0.8) = 0.4 + 0.3 - 0.2 = 0.5
    expected = 0.5
    assert pid.act(_state(e_p=0.4, e_i=0.6, e_d=-0.8)) == pytest.approx(expected)