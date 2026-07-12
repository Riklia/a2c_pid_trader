import pandas as pd
import pytest
import torch

from models.pid_net import PIDNet, PIDNetInputBuilder


def _state(e_p=0.1, e_i=0.5, e_d=-0.05, realized_vol=0.5):
    return pd.Series({
        "e_p": e_p, "e_i": e_i, "e_d": e_d, "realized_vol": realized_vol,
    })


def test_output_shape_matches_three_gains():
    net = PIDNet()
    x = torch.zeros(net.INPUT_SIZE)
    out = net(x)
    assert out.shape == (net.OUTPUT_SIZE,)


def test_output_respects_gain_bounds():
    net = PIDNet(gain_bounds=2.5)
    x = torch.ones(net.INPUT_SIZE) * 100.0  # saturates tanh at +/-1
    out = net(x)
    assert (out.abs() <= 2.5 + 1e-6).all()


def test_zero_gain_bounds_rejected():
    with pytest.raises(ValueError, match="gain_bounds must be positive"):
        PIDNet(gain_bounds=0.0)


def test_near_zero_initialization_produces_near_zero_gains():
    net = PIDNet()
    x = torch.zeros(net.INPUT_SIZE)
    out = net(x)
    assert out.abs().max().item() < 0.05


def test_batch_input_works():
    net = PIDNet()
    batch = torch.zeros(4, net.INPUT_SIZE)
    out = net(batch)
    assert out.shape == (4, net.OUTPUT_SIZE)


def test_input_vector_starts_with_zero_history():
    builder = PIDNetInputBuilder()
    vec = builder.build(_state(e_p=0.1, e_i=0.5, e_d=-0.05))
    # first 4 entries are u(k-1), u(k-2), s(k-1), s(k-2) -- all zero at start
    assert torch.allclose(vec[:4], torch.zeros(4))
    assert vec[4].item() == pytest.approx(0.1)
    assert vec[5].item() == pytest.approx(0.5)
    assert vec[6].item() == pytest.approx(-0.05)


def test_record_shifts_history():
    builder = PIDNetInputBuilder()
    builder.record(position=0.3, realized_vol=0.7)
    builder.record(position=0.5, realized_vol=0.8)

    vec = builder.build(_state())
    # newest first: u(k-1)=0.5, u(k-2)=0.3, s(k-1)=0.8, s(k-2)=0.7
    assert vec[0].item() == pytest.approx(0.5)
    assert vec[1].item() == pytest.approx(0.3)
    assert vec[2].item() == pytest.approx(0.8)
    assert vec[3].item() == pytest.approx(0.7)


def test_only_last_two_values_are_kept():
    builder = PIDNetInputBuilder()
    for i in range(5):
        builder.record(position=float(i), realized_vol=float(i) * 10)

    vec = builder.build(_state())
    assert vec[0].item() == pytest.approx(4.0)
    assert vec[1].item() == pytest.approx(3.0)
    assert vec[2].item() == pytest.approx(40.0)
    assert vec[3].item() == pytest.approx(30.0)


def test_reset_clears_history():
    builder = PIDNetInputBuilder()
    builder.record(position=0.9, realized_vol=0.9)
    builder.reset()
    vec = builder.build(_state())
    assert torch.allclose(vec[:4], torch.zeros(4))


def test_missing_state_field_raises():
    builder = PIDNetInputBuilder()
    bad_state = pd.Series({"e_p": 0.1, "e_i": 0.5})  # missing e_d, realized_vol
    with pytest.raises(ValueError, match="missing required fields"):
        builder.build(bad_state)


def test_builder_output_feeds_net_correctly():
    """Sanity check that shapes line up end-to-end."""
    net = PIDNet()
    builder = PIDNetInputBuilder()

    builder.record(position=0.5, realized_vol=0.6)
    x = builder.build(_state())

    gains = net(x)
    assert gains.shape == (3,)
    assert not torch.isnan(gains).any()
