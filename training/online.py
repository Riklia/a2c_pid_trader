"""Online evaluation: the controller keeps learning while it trades.

This is the adaptive-control mode of the paper: the network is updated bar by
bar from realized outcomes, so it can follow a plant (market) that drifts away
from what it saw in training. A static PID cannot do this.

It is causal by construction. At bar i the controller acts on state i; the loss
of that action needs bar i+1's return, which is known at bar i+1's close, and
the resulting update is applied before the controller acts on bar i+1. No
decision ever depends on a return that has not happened yet.
"""
import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from control.a2c_pid import A2CPID
from environment.trading_env import TradingEnv
from training.train_loop import TrainConfig, train_one_pass


def run_online(env: TradingEnv, controller: A2CPID, config: TrainConfig) -> BacktestResult:
    """Trades `env` once while updating `controller` online, returns the realized result.

    `config.epochs` is forced to 1: a second pass would trade the same bars
    again with knowledge of how they turned out.
    """
    if config.epochs != 1:
        config = TrainConfig(**{**config.__dict__, "epochs": 1})
    reports = train_one_pass(env, controller, config)

    index = env.data.index[: len(env.equity_curve)]
    equity = pd.Series(env.equity_curve, index=index, name="equity")
    return BacktestResult(
        equity=equity,
        positions=pd.Series(env.position_history, index=index, name="position"),
        rewards=pd.Series(np.log(equity / equity.shift(1)).iloc[1:].to_numpy(), index=index[1:], name="reward"),
        actions=pd.Series([r.action for r in reports], index=index[1:], name="action"),
    )
