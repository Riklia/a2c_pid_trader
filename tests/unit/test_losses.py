import pytest
import torch

import math

from training.losses import (
    step_utility, trading_costs, vol_tracking_loss, td_error, actor_loss, critic_loss,
)


def test_step_utility_is_pnl_minus_variance_penalty():
    u = step_utility(torch.tensor(0.5), simple_return=0.02, risk_aversion=4.0)
    # pnl = 0.01, penalty = 0.5 * 4 * 0.01^2 = 0.0002
    assert u.item() == pytest.approx(0.0098)


def test_step_utility_gradient_is_exact():
    position = torch.tensor(0.5, requires_grad=True)
    step_utility(position, simple_return=0.02, risk_aversion=4.0).backward()
    # dU/du = r - risk_aversion * u * r^2
    assert position.grad.item() == pytest.approx(0.02 - 4.0 * 0.5 * 0.02 ** 2)


def test_trading_costs_charge_fee_on_turnover_and_funding_on_position():
    costs = trading_costs(torch.tensor(0.8), prev_position=0.3, funding_rate=0.001, fee_rate=0.0004)
    assert costs.item() == pytest.approx(0.0004 * 0.5 + 0.8 * 0.001)


def test_trading_costs_zero_without_change_or_funding():
    assert trading_costs(torch.tensor(0.4), prev_position=0.4, fee_rate=0.0004).item() == 0.0


def test_vol_tracking_loss_is_minimized_at_target_over_true_vol():
    """Averaged over many bars, the QLIKE minimizer is |u| = target / sigma."""
    sigma, target, n = 0.8, 0.2, 200_000
    returns = torch.randn(n, generator=torch.Generator().manual_seed(0)) * sigma / math.sqrt(8760)
    grid = torch.linspace(0.05, 0.6, 56)
    losses = [vol_tracking_loss(u, returns, target, 8760).mean().item() for u in grid]
    best = grid[int(torch.tensor(losses).argmin())].item()
    assert best == pytest.approx(target / sigma, abs=0.02)


def test_vol_tracking_loss_is_symmetric_in_position_sign():
    a = vol_tracking_loss(torch.tensor(0.5), 0.01, 0.3, 8760)
    b = vol_tracking_loss(torch.tensor(-0.5), 0.01, 0.3, 8760)
    assert a.item() == pytest.approx(b.item())


def test_td_error_matches_definition():
    v_curr = torch.tensor(0.5)
    v_next = torch.tensor(0.8)
    td = td_error(reward=0.1, value_next=v_next, value_current=v_curr, gamma=0.9)
    # 0.1 + 0.9*0.8 - 0.5 = 0.32
    assert td.item() == pytest.approx(0.32)


def test_td_error_preserves_gradient_to_critic():
    v_curr = torch.tensor(0.5, requires_grad=True)
    v_next = torch.tensor(0.8, requires_grad=True)
    td = td_error(reward=0.1, value_next=v_next, value_current=v_curr)
    td.backward()
    assert v_curr.grad is not None
    assert v_next.grad is not None


def test_actor_loss_is_positive_with_nonzero_prediction_error():
    mu = torch.tensor(0.5)
    sigma = torch.tensor(0.1)
    td = torch.tensor(0.2)
    loss = actor_loss(predicted_next_state=mu, actual_next_state=0.3,
                       sigma=sigma, td=td)
    assert loss.item() > 0


def test_actor_loss_prediction_term_scales_with_td_magnitude():
    mu = torch.tensor(0.5)
    sigma = torch.tensor(0.1)
    small_td = torch.tensor(0.01)
    big_td = torch.tensor(1.0)

    loss_small = actor_loss(predicted_next_state=mu, actual_next_state=0.3,
                              sigma=sigma, td=small_td, w_entropy=0.0)
    loss_big = actor_loss(predicted_next_state=mu, actual_next_state=0.3,
                            sigma=sigma, td=big_td, w_entropy=0.0)
    assert loss_big.item() > loss_small.item()


def test_actor_loss_prediction_term_never_fully_collapses_thanks_to_eta():
    mu = torch.tensor(0.5)
    sigma = torch.tensor(0.01)
    td = torch.tensor(0.0)
    loss = actor_loss(predicted_next_state=mu, actual_next_state=0.3,
                       sigma=sigma, td=td, w_prediction=1.0, w_entropy=0.0, eta=1e-3)
    # prediction error = 0.04, eta=1e-3 -> loss should be 4e-5, not zero
    assert loss.item() == pytest.approx(0.04 * 1e-3)


def test_actor_loss_entropy_term_grows_with_sigma():
    mu = torch.tensor(0.5)
    td = torch.tensor(0.2)

    loss_small_sigma = actor_loss(predicted_next_state=mu, actual_next_state=0.5,
                                    sigma=torch.tensor(0.01), td=td,
                                    w_prediction=0.0, w_entropy=1.0)
    loss_big_sigma = actor_loss(predicted_next_state=mu, actual_next_state=0.5,
                                  sigma=torch.tensor(1.0), td=td,
                                  w_prediction=0.0, w_entropy=1.0)
    assert loss_big_sigma.item() > loss_small_sigma.item()


def test_actor_loss_preserves_gradient():
    mu = torch.tensor(0.5, requires_grad=True)
    sigma = torch.tensor(0.1, requires_grad=True)
    td = torch.tensor(0.2)
    loss = actor_loss(predicted_next_state=mu, actual_next_state=0.3,
                       sigma=sigma, td=td)
    loss.backward()
    assert mu.grad is not None
    assert sigma.grad is not None


def test_critic_loss_is_squared_td():
    td = torch.tensor(0.5)
    loss = critic_loss(td, weight=1.0)
    assert loss.item() == pytest.approx(0.25)


def test_critic_loss_zero_when_td_zero():
    assert critic_loss(torch.tensor(0.0)).item() == 0.0


def test_critic_loss_preserves_gradient():
    td = torch.tensor(0.5, requires_grad=True)
    loss = critic_loss(td)
    loss.backward()
    assert td.grad is not None