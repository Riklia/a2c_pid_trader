import numpy as np
import pytest
import torch

from control.vol_target import PIDErrors
from data.model_features import DEFAULT_FEATURES
from models.pid_net import PIDNet, PIDNetInputBuilder


FEATURES = np.array([0.1, 0.5, -0.05])


def test_output_shape_matches_three_gains():
    net = PIDNet()
    x = torch.zeros(net.input_size)
    out = net(x)
    assert out.shape == (net.OUTPUT_SIZE,)


def test_output_respects_gain_bounds():
    net = PIDNet(gain_bounds=2.5)
    x = torch.ones(net.input_size) * 100.0  # saturates tanh at +/-1
    out = net(x)
    assert (out.abs() <= 2.5 + 1e-6).all()


def test_zero_gain_bounds_rejected():
    with pytest.raises(ValueError, match="gain_bounds must be positive"):
        PIDNet(gain_bounds=0.0)


def test_near_zero_initialization_produces_near_zero_gains():
    net = PIDNet()
    x = torch.zeros(net.input_size)
    out = net(x)
    assert out.abs().max().item() < 0.05


def test_batch_input_works():
    net = PIDNet()
    batch = torch.zeros(4, net.input_size)
    out = net(batch)
    assert out.shape == (4, net.OUTPUT_SIZE)


def test_default_input_size_matches_default_features():
    assert PIDNet().input_size == PIDNetInputBuilder.input_size(len(DEFAULT_FEATURES))


def test_input_size_is_configurable():
    net = PIDNet(input_size=11)
    assert net(torch.zeros(11)).shape == (3,)


def test_per_gain_bounds_are_respected():
    net = PIDNet(input_size=4, gain_bounds=(1.0, 0.01, 5.0))
    out = net(torch.ones(4) * 100.0)
    assert (out.abs() <= torch.tensor([1.0, 0.01, 5.0]) + 1e-6).all()


def _errors(e=0.1, e_int=50.0, e_d=-0.05):
    return PIDErrors(e=e, e_int=e_int, e_d=e_d)


def test_input_vector_is_positions_then_law_errors_then_features():
    builder = PIDNetInputBuilder()
    vec = builder.build(_errors(), integral_limit=100.0, features=FEATURES)
    assert vec.shape == (PIDNetInputBuilder.input_size(len(FEATURES)),)
    assert torch.allclose(vec[:2], torch.zeros(2))  # u(k-1), u(k-2) are zero at start
    assert vec[2:5].tolist() == pytest.approx([0.1, 0.5, -0.05])  # e, e_int / limit, e_d
    assert torch.allclose(vec[5:], torch.tensor(FEATURES, dtype=torch.float32))


def test_record_shifts_history():
    builder = PIDNetInputBuilder()
    builder.record(position=0.3)
    builder.record(position=0.5)

    vec = builder.build(_errors(), 100.0, FEATURES)
    # newest first: u(k-1)=0.5, u(k-2)=0.3
    assert vec[0].item() == pytest.approx(0.5)
    assert vec[1].item() == pytest.approx(0.3)


def test_only_last_two_values_are_kept():
    builder = PIDNetInputBuilder()
    for i in range(5):
        builder.record(position=float(i))

    vec = builder.build(_errors(), 100.0, FEATURES)
    assert vec[0].item() == pytest.approx(4.0)
    assert vec[1].item() == pytest.approx(3.0)


def test_reset_clears_history():
    builder = PIDNetInputBuilder()
    builder.record(position=0.9)
    builder.reset()
    vec = builder.build(_errors(), 100.0, FEATURES)
    assert torch.allclose(vec[:2], torch.zeros(2))


def test_builder_output_feeds_net_correctly():
    """Sanity check that shapes line up end-to-end."""
    builder = PIDNetInputBuilder()
    net = PIDNet(input_size=PIDNetInputBuilder.input_size(len(FEATURES)))

    builder.record(position=0.5)
    gains = net(builder.build(_errors(), 100.0, FEATURES))
    assert gains.shape == (3,)
    assert not torch.isnan(gains).any()


def test_four_output_net_supports_bias_and_bounds():
    net = PIDNet(input_size=4, gain_bounds=(0.5, 0.003, 20.0, 0.4))
    assert net.output_size == 4
    assert net.has_bias is True
    out = net(torch.ones(4) * 100.0)
    assert out.shape == (4,)
    assert (out.abs() <= torch.tensor([0.5, 0.003, 20.0, 0.4]) + 1e-6).all()


def test_invalid_gain_bounds_length_rejected():
    with pytest.raises(ValueError, match="length 3/4"):
        PIDNet(input_size=4, gain_bounds=(0.5, 0.5))
