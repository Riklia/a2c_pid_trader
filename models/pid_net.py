from collections import deque

import pandas as pd
import torch
import torch.nn as nn

from data.validation import require_columns


PID_NET_STATE_FIELDS = frozenset({"realized_vol", "e_p", "e_i", "e_d"})


class PIDNet(nn.Module):
    """MLP that outputs dynamic PID gains from recent state history.

    Input: [u(k-1), u(k-2), s(k-1), s(k-2), e_p, e_i, e_d]
        u - previous controller outputs (positions in [-1, 1])
        s - previous realized volatilities
        e_p, e_i, e_d - current PID error signals from the feature table

    Output: [Kp_dyn, Ki_dyn, Kd_dyn], each in [-gain_bounds, +gain_bounds],
    produced by tanh scaled by `gain_bounds`. These are meant to be added to
    static gains from StaticPID, not to replace them.

    Args:
        hidden_sizes: sizes of the hidden layers, all with sigmoid activation
            per the paper's Fig. 2.
        gain_bounds: absolute cap on each dynamic gain. Tanh output is scaled
            by this before returning.
    """

    # u(k-1), u(k-2), s(k-1), s(k-2), e_p, e_i, e_d
    INPUT_SIZE = 7
    # Kp_dyn, Ki_dyn, Kd_dyn
    OUTPUT_SIZE = 3

    def __init__(self, hidden_sizes: tuple[int, ...] = (32, 32), gain_bounds: float = 1.0):
        super().__init__()
        if gain_bounds <= 0:
            raise ValueError("gain_bounds must be positive")

        layers: list[nn.Module] = []
        prev = self.INPUT_SIZE
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.Sigmoid())
            prev = size
        layers.append(nn.Linear(prev, self.OUTPUT_SIZE))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.gain_bounds = gain_bounds

        # weights initialized near zero so early outputs are near zero,
        # matching the paper's "start close to static PID" behavior
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.01, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the MLP and scales tanh output to [-gain_bounds, +gain_bounds]."""
        return self.net(x) * self.gain_bounds


class PIDNetInputBuilder:
    """Assembles the 7-dim input vector for PIDNet from a streaming state row.

    Maintains a length-2 history of past positions and realized volatilities,
    since a single state row only carries the current bar.

    Before the history is full, missing lags are filled with zeros so the
    network can be called from the very first step without special-casing.
    """

    HISTORY_LENGTH = 2

    def __init__(self):
        self._u_hist: deque[float] = deque([0.0] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)
        self._s_hist: deque[float] = deque([0.0] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)

    def build(self, state: pd.Series) -> torch.Tensor:
        """Returns the 7-dim input tensor for the current step."""
        require_columns(state, PID_NET_STATE_FIELDS, "PIDNetInputBuilder.build")
        vec = [
            self._u_hist[-1], self._u_hist[-2],
            self._s_hist[-1], self._s_hist[-2],
            float(state["e_p"]), float(state["e_i"]), float(state["e_d"]),
        ]
        return torch.tensor(vec, dtype=torch.float32)

    def record(self, position: float, realized_vol: float) -> None:
        """Adds the executed position and observed vol to the history."""
        self._u_hist.append(float(position))
        self._s_hist.append(float(realized_vol))

    def reset(self) -> None:
        self._u_hist = deque([0.0] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)
        self._s_hist = deque([0.0] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)