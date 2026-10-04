import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def sharpe_ratio(rewards: pd.Series, bars_per_year: float) -> float:
    mean = rewards.mean()
    std = rewards.std(ddof=1)
    if np.isnan(std) or std < 1e-12:
        return float("nan")
    return float(mean / std * np.sqrt(bars_per_year))


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    return float(drawdown.max())


def vol_tracking_rmse(realized_vol: pd.Series, target_vol: float) -> float:
    diff = realized_vol - target_vol
    return float(np.sqrt((diff ** 2).mean()))

def annualized_vol(rewards: pd.Series, bars_per_year: float) -> float:
    """Realized vol of the strategy's per-bar log returns, annualized."""
    return float(rewards.std(ddof=1) * np.sqrt(bars_per_year))


def rolling_portfolio_vol_rmse(
    rewards: pd.Series,
    target_vol: float,
    window: int,
    bars_per_year: float,
) -> float:
    """RMSE between the strategy's rolling realized vol and target_vol.

    This is the vol-targeting control metric: it measures what the portfolio
    actually experienced, not what the controller intended.

    Args:
        rewards: per-bar log returns of the strategy.
        target_vol: annualized vol the controller was asked to track.
        window: bars per rolling vol estimate.
        bars_per_year: annualization factor.
    """
    rolling = rewards.rolling(window).std() * np.sqrt(bars_per_year)
    diff = rolling.dropna() - target_vol
    return float(np.sqrt((diff ** 2).mean()))


def turnover(positions: pd.Series) -> float:
    """Total absolute position change, in multiples of equity."""
    return float(positions.diff().abs().sum())


def beta_alpha(
    rewards: pd.Series,
    market_returns: pd.Series,
    bars_per_year: float,
) -> tuple[float, float]:
    """OLS beta of strategy returns on market returns, and annualized alpha.

    Separates what a strategy earned by simply being exposed to the market
    (beta) from what it earned on top of that (alpha). A long-only strategy
    that wins in up markets and loses in down markets has high beta and no alpha.

    Args:
        rewards: per-bar log returns of the strategy.
        market_returns: per-bar log returns of the traded asset, aligned on
            the same timestamps.
        bars_per_year: annualization factor for alpha.
    """
    aligned = pd.concat([rewards.rename("s"), market_returns.rename("m")], axis=1).dropna()
    var = aligned["m"].var(ddof=1)
    if np.isnan(var) or var < 1e-18:
        return float("nan"), float("nan")
    beta = aligned["s"].cov(aligned["m"]) / var
    alpha = (aligned["s"].mean() - beta * aligned["m"].mean()) * bars_per_year
    return float(beta), float(alpha)
