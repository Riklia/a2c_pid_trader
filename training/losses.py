import math

import torch


def compute_reward(
    realized_vol: float,
    target_vol: float,
    realized_vol_prev: float,
    target_vol_prev: float,
    action: float,
    action_prev: float = 0.0,
    w_error: float = 1.0,
    w_error_rate: float = 0.0,
    w_action: float = 0.1,
) -> float:
    """Computes Quadratic reward penalizing portfolio-vol tracking error, error rate, and action size.

    The controlled variable in this domain is portfolio-level volatility,
    `portfolio_vol = |position| * realized_vol`, since `realized_vol` itself
    is exogenous and cannot be moved by the controller. Tracking error
    is `target_vol - portfolio_vol`.

    Args:
        realized_vol: current realized volatility of the market.
        target_vol: current target portfolio volatility.
        realized_vol_prev: previous bar's realized volatility.
        target_vol_prev: previous bar's target volatility.
        action: controller output at this step, in [-1, 1].
        action_prev: controller output at the previous step, used for error_rate.
        w_error: weight on squared tracking error.
        w_error_rate: weight on squared change of the tracking error.
        w_action: weight on squared action, discourages excessive turnover.
    """
    portfolio_vol = abs(action) * realized_vol
    portfolio_vol_prev = abs(action_prev) * realized_vol_prev

    error = target_vol - portfolio_vol
    error_prev = target_vol_prev - portfolio_vol_prev
    error_rate = error - error_prev

    return -(w_error * error ** 2 + w_error_rate * error_rate ** 2 + w_action * action ** 2)


def td_error(
    reward: float,
    value_next: torch.Tensor,
    value_current: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Computes the temporal-difference error.

    delta = R + gamma * V(s_{k+1}) - V(s_k)

    Args:
        reward: scalar reward at this step.
        value_next: critic estimate at the next state (differentiable).
        value_current: critic estimate at the current state (differentiable).
        gamma: discount factor.
    """
    return reward + gamma * value_next - value_current


def actor_loss(
    predicted_next_state: torch.Tensor,
    actual_next_state: float,
    sigma: torch.Tensor,
    td: torch.Tensor,
    w_prediction: float = 1.0,
    w_entropy: float = 0.01,
    eta: float = 1e-3,
) -> torch.Tensor:
    """
    L_alpha = w1 * (s_m - s)^2 * (eta + |delta_TD|) + w2 * sqrt(2*pi*e) * sigma^2

    The first term is prediction error, scaled by |TD| so exploration continues
    while the critic is still uncertain. eta prevents the whole term from
    collapsing to zero when TD is near zero. The second term is a Gaussian
    entropy bonus that pushes sigma up so the actor keeps exploring.
    """
    prediction_term = (predicted_next_state - actual_next_state) ** 2 * (eta + td.abs())
    entropy_term = math.sqrt(2 * math.pi * math.e) * sigma ** 2
    return w_prediction * prediction_term + w_entropy * entropy_term


def critic_loss(td: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Critic loss: L_c = w3 * delta_TD^2."""
    return weight * td ** 2
