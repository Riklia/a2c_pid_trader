import math

import torch


def step_utility(
    position: torch.Tensor,
    simple_return: float,
    risk_aversion: float = 2.0,
) -> torch.Tensor:
    """Mean-variance utility of holding `position` over one bar, before costs.

        U = u * r - (risk_aversion / 2) * (u * r)^2

    Differentiable in `position`. The market return is exogenous, so the
    gradient through u is exact and needs no critic. Under this utility the
    unconstrained optimum is u* = mu / (risk_aversion * sigma^2).
    """
    pnl = position * simple_return
    return pnl - 0.5 * risk_aversion * pnl ** 2


def trading_costs(
    position: torch.Tensor,
    prev_position: float,
    funding_rate: float = 0.0,
    fee_rate: float = 0.0004,
) -> torch.Tensor:
    """Fee on turnover plus funding paid on this bar, as a fraction of equity.

    Mirrors what TradingEnv charges.
    """
    return fee_rate * (position - prev_position).abs() + position * funding_rate


def vol_tracking_loss(
    position: torch.Tensor,
    simple_return: float,
    target_vol: float,
    bars_per_year: float,
) -> torch.Tensor:
    """QLIKE loss for tracking target_vol with exposure |position|.

        L = (u * r)^2 * bars_per_year / target_vol^2 - log(u^2)

    In expectation L is minimized exactly at |u| = target_vol / sigma, where
    sigma is the true annualized vol of the bar. Unlike a squared error on
    (|u| * sigma - target), it needs no vol estimate as a label: the realized
    squared return is an unbiased variance sample. Vol is far more predictable
    than returns, so this term has a much higher signal-to-noise ratio than
    the utility. Undefined at u = 0; callers skip flat bars.
    """
    variance_ratio = (position * simple_return) ** 2 * bars_per_year / target_vol ** 2
    return variance_ratio - torch.log(position ** 2)


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
