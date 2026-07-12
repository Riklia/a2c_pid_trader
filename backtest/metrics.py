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

def portfolio_vol_tracking_rmse(
    positions: pd.Series,
    realized_vol: pd.Series,
    target_vol: float,
) -> float:
    """Root mean squared deviation of |position| * realized_vol from target_vol.

    This is the correct volatility-targeting control metric for the trading:
    realized_vol is exogenous, so the controller can only influence the
    exposure, and portfolio-level vol is |position| * realized_vol.
    """
    aligned = pd.concat([positions.rename("pos"), realized_vol.rename("vol")], axis=1).dropna()
    portfolio_vol = aligned["pos"].abs() * aligned["vol"]
    diff = portfolio_vol - target_vol
    return float(np.sqrt((diff ** 2).mean()))
