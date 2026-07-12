from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from control.static_pid import StaticPID
from data.scaling import RollingZScoreScaler
from data.validation import require_columns
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet, PIDNetInputBuilder, PID_NET_STATE_FIELDS


A2C_PID_STATE_FIELDS = PID_NET_STATE_FIELDS

# features that get z-scored before being fed to PIDNet / Actor / Critic
_SCALED_FEATURES = ["realized_vol", "e_p", "e_i", "e_d"]


@dataclass
class A2CPID:
    """Self-tuning PID controller with actor-critic-driven online gain adaptation.

    The final action is:
        u = clip(u_static + u_dynamic, -1, 1)
        u_static = StaticPID(kp_s, ki_s, kd_s)  [e_p, e_i, e_d]
        u_dynamic = PIDNet(scaled_state)  [e_p, e_i, e_d]

    Actor and Critic are used only during training via `observe`, they do
    not affect `act` output directly.

    Attributes:
        static_pid: fixed-gain baseline controller.
        pid_net: network producing dynamic gains added to the static ones.
        actor: predicts next realized_vol, used for pid_net gradient.
        critic: predicts V(state), used for TD error.
        scaler: online z-score normalization for state features.
        input_builder: keeps 2-step history of u and s for the pid_net input.
        position_bounds: (low, high) clip range for the final action.
    """

    static_pid: StaticPID
    pid_net: PIDNet
    actor: Actor
    critic: Critic
    scaler: RollingZScoreScaler = field(
        default_factory=lambda: RollingZScoreScaler(_SCALED_FEATURES)
    )
    input_builder: PIDNetInputBuilder = field(default_factory=PIDNetInputBuilder)
    position_bounds: tuple[float, float] = (-1.0, 1.0)

    # Populated during act() so observe() knows what happened at this step
    _last_pid_input: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_action: float | None = field(default=None, init=False, repr=False)
    _last_scaled_state: pd.Series | None = field(default=None, init=False, repr=False)

    def act(self, state: pd.Series) -> float:
        """Returns a target position, running PIDNet in inference mode.

        Uses raw e_p/e_i/e_d for the PID formula and scaled features only as
        input to the network: the linear PID law needs its inputs in original
        units to give a meaningful position, while the network learns better
        on normalized inputs.
        """
        require_columns(state, A2C_PID_STATE_FIELDS, "A2CPID.act")

        scaled = self.scaler.transform(state)
        pid_input = self.input_builder.build(scaled)

        with torch.no_grad():
            dyn_gains = self.pid_net(pid_input)
        kp_d, ki_d, kd_d = (float(g) for g in dyn_gains)

        kp = self.static_pid.kp + kp_d
        ki = self.static_pid.ki + ki_d
        kd = self.static_pid.kd + kd_d
        raw = kp * state["e_p"] + ki * state["e_i"] + kd * state["e_d"]

        low, high = self.position_bounds
        action = float(np.clip(raw, low, high))

        self._last_pid_input = pid_input
        self._last_action = action
        self._last_scaled_state = scaled

        return action

    def record_observation(self, position: float, realized_vol: float) -> None:
        """Updates the input builder's history after env.step has been executed."""
        self.input_builder.record(position=position, realized_vol=realized_vol)

    def build_training_inputs(self) -> tuple[torch.Tensor, float]:
        """Returns the (pid_input, action) captured during the last act() call.

        Meant for the training loop, which needs these to compute losses
        against the observed next state and reward.
        """
        if self._last_pid_input is None or self._last_action is None:
            raise RuntimeError("build_training_inputs called before any act()")
        return self._last_pid_input, self._last_action
