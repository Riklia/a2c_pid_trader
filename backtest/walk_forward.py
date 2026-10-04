from dataclasses import dataclass

import pandas as pd

from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import (
    total_return, sharpe_ratio, max_drawdown, annualized_vol,
    rolling_portfolio_vol_rmse, turnover, beta_alpha,
)
from control.controller_base import Controller
from environment.trading_env import TradingEnv


@dataclass
class WalkForwardFold:
    fold_id: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_data.index[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_data.index[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_data.index[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_data.index[-1]


def make_walk_forward_folds(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardFold]:
    """Yields sliding train/test window pairs over an ordered dataset.

    Args:
        data: full feature table, DatetimeIndex, sorted ascending.
        train_size: number of bars per train window.
        test_size: number of bars per test window.
        step_size: how far to slide between folds; defaults to test_size
            for non-overlapping test windows.
    """
    if step_size is None:
        step_size = test_size

    folds = []
    fold_id = 0
    start = 0
    while start + train_size + test_size <= len(data):
        train_data = data.iloc[start : start + train_size]
        test_data = data.iloc[start + train_size : start + train_size + test_size]
        folds.append(WalkForwardFold(fold_id=fold_id, train_data=train_data, test_data=test_data))
        start += step_size
        fold_id += 1
    return folds


@dataclass
class FoldMetrics:
    """Metrics for one controller on one fold's test window."""
    fold_id: int
    controller_name: str
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    total_return: float
    sharpe: float
    max_drawdown: float
    ann_vol: float
    vol_rmse: float
    turnover: float
    beta: float
    alpha: float
    seed: int | None = None


def evaluate_on_test(
    fold: WalkForwardFold,
    controller: Controller,
    controller_name: str,
    target_vol: float,
    fee_rate: float,
    bars_per_year: float,
    max_position: float = 1.0,
    vol_window: int = 168,
    seed: int | None = None,
) -> tuple[FoldMetrics, BacktestResult]:
    """Runs `controller` on fold.test_data, returns metrics and the raw result.

    vol_window is the number of bars per rolling estimate in vol_rmse.
    """
    env = TradingEnv(fold.test_data, fee_rate=fee_rate, max_position=max_position)
    result = run_backtest(env, controller)
    beta, alpha = beta_alpha(result.rewards, fold.test_data["log_return"], bars_per_year)

    metrics = FoldMetrics(
        fold_id=fold.fold_id,
        controller_name=controller_name,
        train_start=fold.train_start,
        test_start=fold.test_start,
        test_end=fold.test_end,
        total_return=total_return(result.equity),
        sharpe=sharpe_ratio(result.rewards, bars_per_year=bars_per_year),
        max_drawdown=max_drawdown(result.equity),
        ann_vol=annualized_vol(result.rewards, bars_per_year),
        vol_rmse=rolling_portfolio_vol_rmse(result.rewards, target_vol, vol_window, bars_per_year),
        turnover=turnover(result.positions),
        beta=beta,
        alpha=alpha,
        seed=seed,
    )
    return metrics, result
