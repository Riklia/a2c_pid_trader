from dataclasses import dataclass

import pandas as pd

from control.controller_base import Controller
from environment.trading_env import TradingEnv


@dataclass
class BacktestResult:
    """The result produced by a backtest.

    Attributes:
        equity: portfolio value at each bar.
        positions: position held going into each bar.
        rewards: log change in equity at each step.
        actions: raw controller output at each step.

    All the attributes are indexed by bar timestamp.
    """
    equity: pd.Series
    positions: pd.Series
    rewards: pd.Series
    actions: pd.Series


def run_backtest(env: TradingEnv, controller: Controller) -> BacktestResult:
    """Steps the env with the controller's actions until the episode ends.

    The controller sees the state at bar i and returns a target position
    that will be applied against bar i+1's return, matching TradingEnv
    no-look-ahead convention.

    Controllers that keep a history of their own observations (A2CPID) are
    fed the executed position and the next bar's realized vol after every
    step, exactly as the training loop does. Without this their history
    would stay frozen at its last training value for the whole backtest.

    Args:
        env: fresh environment, already constructed but not yet stepped.
        controller: any object satisfying the Controller protocol.
    """
    env.reset()

    timestamps: list[pd.Timestamp] = []
    rewards: list[float] = []
    actions: list[float] = []

    while not env.done:
        state = env.data.iloc[env.current_step]
        action = controller.act(state)
        next_row, reward, _ = env.step(action)
        if hasattr(controller, "record_observation"):
            controller.record_observation(position=action, realized_vol=float(next_row["realized_vol"]))

        timestamps.append(env.data.index[env.current_step])
        rewards.append(reward)
        actions.append(action)

    equity_index = env.data.index[: len(env.equity_curve)]
    return BacktestResult(
        equity=pd.Series(env.equity_curve, index=equity_index, name="equity"),
        positions=pd.Series(env.position_history, index=equity_index, name="position"),
        rewards=pd.Series(rewards, index=timestamps, name="reward"),
        actions=pd.Series(actions, index=timestamps, name="action"),
    )