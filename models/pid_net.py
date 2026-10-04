from collections import deque

import numpy as np
import torch
import torch.nn as nn

from control.vol_target import PIDErrors
from data.model_features import DEFAULT_FEATURES


class PIDNet(nn.Module):
    """MLP that outputs dynamic PID gains from recent state history.

    Input: [u(k-1), u(k-2), e, e_int, e_d, f_1, ..., f_n], built by PIDNetInputBuilder
        u - the controller's two previous positions
        e, e_int, e_d - the control law's own error signals (see VolTargetLaw),
            the quantities the gains multiply
        f - the normalized model features configured in a2c_pid.features,
            see data.model_features

    Output: [Kp_dyn, Ki_dyn, Kd_dyn], each in [-bound, +bound], produced by
    tanh scaled by `gain_bounds`. These are meant to be added to static
    gains from StaticPID, not to replace them.

    Args:
        input_size: PIDNetInputBuilder.input_size(n_features). The default
            matches DEFAULT_FEATURES.
        hidden_sizes: sizes of the hidden layers, all with sigmoid activation
            per the paper's Fig. 2.
        gain_bounds: absolute cap on each dynamic gain, either one value for
            all three or a (kp, ki, kd) triple, since the gains live on very
            different scales. Tanh output is scaled by this before returning.
    """

    # Kp_dyn, Ki_dyn, Kd_dyn
    OUTPUT_SIZE = 3

    def __init__(
            self,
            input_size: int | None = None,
            hidden_sizes: tuple[int, ...] = (32, 32),
            gain_bounds: float | tuple[float, float, float] = 1.0,
    ):
        super().__init__()
        if input_size is None:
            input_size = PIDNetInputBuilder.input_size(len(DEFAULT_FEATURES))
        self.input_size = input_size
        bounds = torch.as_tensor(gain_bounds, dtype=torch.float32).expand(self.OUTPUT_SIZE).clone()
        if (bounds <= 0).any():
            raise ValueError("gain_bounds must be positive")

        layers: list[nn.Module] = []
        prev = self.input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.Sigmoid())
            prev = size
        layers.append(nn.Linear(prev, self.OUTPUT_SIZE))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.register_buffer("gain_bounds", bounds)

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
    """Assembles the PIDNet input: previous positions, law errors, then the features.

    The features arrive already normalized (see data.model_features). The
    integral error is divided by the law's integral limit so it lies in
    [-1, 1]; e and e_d are dimensionless log errors already. Before any
    position is recorded the position history is zeros, so the network can
    be called from the very first step without special-casing.
    """

    HISTORY_LENGTH = 2
    N_ERRORS = 3

    def __init__(self):
        self._u_hist: deque[float] = deque([0.0] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)

    @classmethod
    def input_size(cls, n_features: int) -> int:
        return cls.HISTORY_LENGTH + cls.N_ERRORS + n_features

    def build(self, errors: PIDErrors, integral_limit: float, features: np.ndarray) -> torch.Tensor:
        """Returns [u(k-1), u(k-2), e, e_int / integral_limit, e_d, *features] as a float tensor."""
        vec = [
            self._u_hist[-1], self._u_hist[-2],
            errors.e, errors.e_int / integral_limit, errors.e_d,
            *np.asarray(features, dtype=np.float64),
        ]
        return torch.tensor(vec, dtype=torch.float32)

    def record(self, position: float) -> None:
        """Adds the executed position to the history."""
        self._u_hist.append(float(position))

    def reset(self) -> None:
        self._u_hist = deque([0.0] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)
