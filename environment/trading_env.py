import numpy as np
import pandas as pd

from data.validation import require_columns


class TradingEnv:
    """Bar-by-bar trading simulator over a precomputed feature table.

    A position submitted at step i is exposed to bar i+1's return, not bar
    i's own return. This is because bar i's log_return is already a known fact
    by the time the controller acts on row i, so applying it to the same decision
    would be look-ahead.

    Fees are charged immediately on turnover, and funding is charged only on bars
    flagged as a real settlement.

    Args:
        data: feature table indexed by bar timestamp, must contain columns
            close, log_return, funding, and is_funding_settlement.
        fee_rate: taker fee charged on turnover as a fraction of equity.
        initial_equity: starting portfolio value.
    """

    REQUIRED_COLUMNS = {"close", "log_return", "funding", "is_funding_settlement"}

    def __init__(self, data: pd.DataFrame, fee_rate: float = 0.0004, initial_equity: float = 1.0):
        require_columns(data, self.REQUIRED_COLUMNS, "TradingEnv")
        if len(data) < 2:
            raise ValueError("data must have at least 2 rows to step through")

        self.data = data
        self.fee_rate = fee_rate
        self.initial_equity = initial_equity
        self.reset()

    def reset(self) -> pd.Series:
        """Resets equity and position to initial state, returns the first row."""
        self._step_idx = 0
        self.equity = self.initial_equity
        self.position = 0.0
        self.equity_curve = [self.equity]
        self.position_history = [self.position]
        return self.data.iloc[self._step_idx]

    @property
    def done(self) -> bool:
        return self._step_idx >= len(self.data) - 1

    def step(self, target_position: float):
        """Applies a target position, charges fees on turnover, then earns
        the next bar's return and any funding due on that bar.

        Args:
            target_position: desired fraction of equity in the position,
                clipped to [-1, 1] (negative = short, positive = long).

        Returns:
            (next_row, reward, done). reward is the log change in equity
            over this step.
        """
        if self.done:
            raise RuntimeError("step called after the episode already ended, call reset()")

        target_position = float(np.clip(target_position, -1.0, 1.0))
        next_row = self.data.iloc[self._step_idx + 1]
        equity_before = self.equity

        turnover = abs(target_position - self.position)
        self.equity -= turnover * self.fee_rate * self.equity
        self.position = target_position

        self.equity += self.position * next_row["log_return"] * self.equity

        if next_row["is_funding_settlement"]:
            self.equity -= self.position * next_row["funding"] * self.equity

        reward = np.log(self.equity / equity_before) if equity_before > 0 else -np.inf

        self._step_idx += 1
        self.equity_curve.append(self.equity)
        self.position_history.append(self.position)

        return next_row, reward, self.done