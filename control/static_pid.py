from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StaticPID:
    """Fixed-gain PID over pre-computed error signals in the feature table.

    The state row must contain e_p, e_i, e_d columns produced by
    features.compute_pid_errors. The output is the target position, clipped
    to [-1, 1] to match TradingEnv action space.

    Attributes:
        kp: proportional gain.
        ki: integral gain.
        kd: derivative gain.
        position_bounds: (low, high) clip range for the output.
    """
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    position_bounds: tuple[float, float] = (-1.0, 1.0)

    def act(self, state: pd.Series) -> float:
        raw = self.kp * state["e_p"] + self.ki * state["e_i"] + self.kd * state["e_d"]
        low, high = self.position_bounds
        return float(np.clip(raw, low, high))