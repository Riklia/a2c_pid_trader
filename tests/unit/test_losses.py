import pytest
import torch

from training.losses import compute_reward, td_error, actor_loss, critic_loss


def test_zero_error_zero_action_gives_zero_reward():
    r = compute_reward(
        realized_vol=1.0, target_vol=0.5,
        realized_vol_prev=1.0, target_vol_prev=0.5,
        action=0.5, action_prev=0.5,
        w_error=1.0, w_error_rate=1.0, w_action=0.0,
    )
    assert r == pytest.approx(0.0)


def test_reward_is_always_nonpositive():
    r = compute_reward(realized_vol=0.7, target_vol=0.5,
                        realized_vol_prev=0.6, target_vol_prev=0.5,
                        action=0.3)
    assert r <= 0.0


def test_zero_position_gives_full_target_as_error():
    r = compute_reward(
        realized_vol=0.7, target_vol=0.5,
        realized_vol_prev=0.7, target_vol_prev=0.5,
        action=0.0, w_error=1.0, w_error_rate=0.0, w_action=0.0,
    )
    # error = 0.5 - 0*0.7 = 0.5 -> squared = 0.25
    assert r == pytest.approx(-0.25)


def test_portfolio_vol_at_target_gives_zero_error_term():
    r = compute_reward(
        realized_vol=1.0, target_vol=0.5,
        realized_vol_prev=1.0, target_vol_prev=0.5,
        action=0.5, w_error=1.0, w_error_rate=0.0, w_action=0.0,
    )
    # portfolio_vol = 0.5*1.0 = 0.5 = target -> error = 0
    assert r == pytest.approx(0.0)


def test_short_position_still_reduces_error():
    r_long = compute_reward(
        realized_vol=1.0, target_vol=0.5,
        realized_vol_prev=1.0, target_vol_prev=0.5,
        action=0.5, w_error=1.0, w_error_rate=0.0, w_action=0.0,
    )
    r_short = compute_reward(
        realized_vol=1.0, target_vol=0.5,
        realized_vol_prev=1.0, target_vol_prev=0.5,
        action=-0.5, w_error=1.0, w_error_rate=0.0, w_action=0.0,
    )
    assert r_long == pytest.approx(r_short)


def test_reward_action_term_matches_formula():
    r = compute_reward(realized_vol=0.5, target_vol=0.5,
                        realized_vol_prev=0.5, target_vol_prev=0.5,
                        action=0.5, w_error=0.0, w_error_rate=0.0, w_action=1.0)
    assert r == pytest.approx(-0.25)


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