import pandas as pd
import pytest

from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from control.controller_base import Controller
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet


def _state(e_p=0.1, e_i=0.5, e_d=-0.05, realized_vol=0.5):
    return pd.Series({
        "e_p": e_p, "e_i": e_i, "e_d": e_d, "realized_vol": realized_vol,
    })


def _fresh_controller(kp=1.0, ki=0.0, kd=0.0):
    return A2CPID(
        static_pid=StaticPID(kp=kp, ki=ki, kd=kd),
        pid_net=PIDNet(),
        actor=Actor(),
        critic=Critic(),
    )


def test_act_returns_float_in_bounds():
    ctrl = _fresh_controller()
    action = ctrl.act(_state())
    assert isinstance(action, float)
    assert -1.0 <= action <= 1.0


def test_missing_state_field_raises():
    ctrl = _fresh_controller()
    with pytest.raises(ValueError, match="A2CPID.act.*missing required fields"):
        ctrl.act(pd.Series({"e_p": 0.1, "e_i": 0.5}))  # e_d, realized_vol missing


def test_output_is_clipped_to_position_bounds():
    ctrl = A2CPID(
        static_pid=StaticPID(kp=100.0),  # huge static kp forces saturation
        pid_net=PIDNet(),
        actor=Actor(),
        critic=Critic(),
    )
    action = ctrl.act(_state(e_p=0.5))
    assert action == 1.0


def test_near_zero_pid_net_start_matches_static_pid_output():
    """With small pid_net init, first action should be close to StaticPID's."""
    ctrl = _fresh_controller(kp=1.0)
    static_only = StaticPID(kp=1.0)

    state = _state(e_p=0.3, e_i=0.0, e_d=0.0)
    a_full = ctrl.act(state)
    a_static = static_only.act(state)

    assert a_full == pytest.approx(a_static, abs=0.05)


def test_record_observation_advances_history_seen_by_next_act():
    ctrl = _fresh_controller()
    state = _state(e_p=0.2, e_i=0.0, e_d=0.0, realized_vol=0.7)

    ctrl.act(state)
    ctrl.record_observation(position=0.3, realized_vol=0.65)
    ctrl.act(state)

    pid_input, _ = ctrl.build_training_inputs()
    # after one record: u(k-1)=0.3, s(k-1)=0.65
    assert pid_input[0].item() == pytest.approx(0.3)
    assert pid_input[2].item() == pytest.approx(0.65)


def test_build_training_inputs_before_any_act_raises():
    ctrl = _fresh_controller()
    with pytest.raises(RuntimeError, match="before any act"):
        ctrl.build_training_inputs()


def test_satisfies_controller_protocol():
    from control.controller_base import Controller
    ctrl = _fresh_controller()
    assert isinstance(ctrl, Controller)


def test_act_is_pure_inference_no_gradient_captured():
    ctrl = _fresh_controller()
    ctrl.act(_state())
    pid_input, _ = ctrl.build_training_inputs()
    assert not pid_input.requires_grad
