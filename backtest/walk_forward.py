from dataclasses import dataclass

import pandas as pd

from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import (
    total_return, sharpe_ratio, max_drawdown, portfolio_vol_tracking_rmse,
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
    vol_rmse: float


def evaluate_on_test(
    fold: WalkForwardFold,
    controller: Controller,
    controller_name: str,
    target_vol: float,
    fee_rate: float,
    bars_per_year: float,
) -> tuple[FoldMetrics, BacktestResult]:
    """Runs `controller` on fold.test_data, returns metrics and the raw result."""
    env = TradingEnv(fold.test_data, fee_rate=fee_rate)
    result = run_backtest(env, controller)

    metrics = FoldMetrics(
        fold_id=fold.fold_id,
        controller_name=controller_name,
        train_start=fold.train_start,
        test_start=fold.test_start,
        test_end=fold.test_end,
        total_return=total_return(result.equity),
        sharpe=sharpe_ratio(result.rewards, bars_per_year=bars_per_year),
        max_drawdown=max_drawdown(result.equity),
        vol_rmse=portfolio_vol_tracking_rmse(result.positions, fold.test_data["realized_vol"], target_vol),
    )
    return metrics, result