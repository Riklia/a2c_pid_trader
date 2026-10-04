import pandas as pd
import pytest
import torch

from control.a2c_pid import A2CPID
from control.controller_base import Controller
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw
from data.model_features import FeatureNormalizer, FeatureSpec
from models.pid_net import PIDNet, PIDNetInputBuilder


def _state(log_return=0.001, vol_forecast=0.5, realized_vol=0.6, **extra):
    return pd.Series({"log_return": log_return, "vol_forecast": vol_forecast, "realized_vol": realized_vol, **extra})


def _fresh_controller(kp=1.0, ki=0.0, kd=0.0, target_vol=0.25, max_position=1.0, specs=None):
    torch.manual_seed(0)
    law = VolTargetLaw(target_vol=target_vol, max_position=max_position)
    kwargs = {}
    n_features = 2  # DEFAULT_FEATURES
    if specs is not None:
        kwargs["normalizer"] = FeatureNormalizer(specs)
        n_features = len(specs)
    return A2CPID(
        static_pid=StaticPID(law=law, kp=kp, ki=ki, kd=kd),
        pid_net=PIDNet(input_size=PIDNetInputBuilder.input_size(n_features)),
        **kwargs,
    )


def test_act_returns_float_in_bounds():
    ctrl = _fresh_controller()
    action = ctrl.act(_state())
    assert isinstance(action, float)
    assert 0.0 <= action <= 1.0


def test_missing_state_field_raises():
    ctrl = _fresh_controller()
    with pytest.raises(ValueError, match="A2CPID.act.*missing required fields"):
        ctrl.act(pd.Series({"log_return": 0.0, "vol_forecast": 0.5}))  # realized_vol missing


def test_output_is_capped():
    ctrl = _fresh_controller(target_vol=1.0)
    assert ctrl.act(_state(vol_forecast=0.1)) == 1.0


def test_fresh_network_starts_close_to_its_static_pid():
    ctrl = _fresh_controller(kp=1.0)
    static = StaticPID(law=VolTargetLaw(target_vol=0.25), kp=1.0)
    state = _state(vol_forecast=0.5)
    assert ctrl.act(state) == pytest.approx(static.act(state), abs=0.01)


def test_differentiable_action_matches_executed_action():
    ctrl = _fresh_controller()
    action = ctrl.act(_state())
    position, dyn = ctrl.differentiable_action()
    assert position.item() == pytest.approx(action, rel=1e-6)
    assert dyn.shape == (3,)


def test_differentiable_action_carries_gradient_into_pid_net():
    ctrl = _fresh_controller()
    ctrl.act(_state(vol_forecast=0.4))
    position, _ = ctrl.differentiable_action()
    position.backward()
    grads = [p.grad for p in ctrl.pid_net.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_differentiable_action_before_any_act_raises():
    ctrl = _fresh_controller()
    with pytest.raises(RuntimeError, match="before any act"):
        ctrl.differentiable_action()


def test_act_records_its_own_position_history():
    ctrl = _fresh_controller()
    first = ctrl.act(_state())
    ctrl.act(_state())
    assert ctrl._last_pid_input[0].item() == pytest.approx(first)


def test_network_sees_law_errors_then_the_configured_features():
    specs = [FeatureSpec("funding", norm="none"), FeatureSpec("realized_vol", symbol="ETHUSDT", norm="none")]
    ctrl = _fresh_controller(specs=specs)
    ctrl.act(_state(vol_forecast=0.5, funding=0.0002, **{"ETHUSDT:realized_vol": 0.9}))

    pid_input = ctrl._last_pid_input
    ctx = ctrl._last_ctx
    assert pid_input[2].item() == pytest.approx(ctx.errors.e)  # log(0.25 / 0.5)
    assert pid_input[5:].tolist() == pytest.approx([0.0002, 0.9])


def test_missing_cross_asset_feature_raises():
    ctrl = _fresh_controller(specs=[FeatureSpec("realized_vol", symbol="ETHUSDT")])
    with pytest.raises(ValueError, match="ETHUSDT:realized_vol"):
        ctrl.act(_state())


def test_network_input_size_must_match_feature_count():
    with pytest.raises(ValueError, match="input_size=7"):
        A2CPID(
            static_pid=StaticPID(law=VolTargetLaw(target_vol=0.25)),
            pid_net=PIDNet(input_size=11),
            normalizer=FeatureNormalizer([FeatureSpec("e_p"), FeatureSpec("e_i")]),
        )


def test_reset_keeps_weights_but_clears_episode_state():
    ctrl = _fresh_controller()
    for _ in range(3):
        ctrl.act(_state())
    weights = [p.detach().clone() for p in ctrl.pid_net.parameters()]
    ctrl.reset()
    ctrl.act(_state())
    assert ctrl._last_pid_input[0].item() == 0.0
    assert all(torch.equal(a, b) for a, b in zip(weights, ctrl.pid_net.parameters()))


def test_satisfies_controller_protocol():
    assert isinstance(_fresh_controller(), Controller)


def test_act_is_pure_inference_no_gradient_captured():
    ctrl = _fresh_controller()
    ctrl.act(_state())
    assert not ctrl._last_pid_input.requires_grad
