from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from control.static_pid import REQUIRED_STATE_FIELDS as PID_LAW_FIELDS, StaticPID
from data.model_features import DEFAULT_FEATURES, FeatureNormalizer
from data.validation import require_columns
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet, PIDNetInputBuilder


@dataclass
class A2CPID:
    """Self-tuning PID controller with actor-critic-driven online gain adaptation.

    The final action is:
        u = clip((kp_s + kp_d) * e_p + (ki_s + ki_d) * e_i + (kd_s + kd_d) * e_d, -1, 1)
        [kp_d, ki_d, kd_d] = PIDNet([u(k-1), u(k-2), normalized model features])

    The PID law always uses the raw e_p, e_i, e_d columns. The network sees
    whatever features `normalizer` is configured with (see data.model_features).

    Actor and Critic are used only during training, they do not affect `act`
    output directly.

    Attributes:
        static_pid: fixed-gain baseline controller.
        pid_net: network producing dynamic gains added to the static ones. Its
            input_size must match the normalizer's feature count.
        actor: predicts next realized_vol, used for pid_net gradient.
        critic: predicts V(state), used for TD error.
        normalizer: selects and normalizes the model features.
        input_builder: keeps the 2-step position history for the pid_net input.
        position_bounds: (low, high) clip range for the final action.
    """

    static_pid: StaticPID
    pid_net: PIDNet
    actor: Actor
    critic: Critic
    normalizer: FeatureNormalizer = field(default_factory=lambda: FeatureNormalizer(DEFAULT_FEATURES))
    input_builder: PIDNetInputBuilder = field(default_factory=PIDNetInputBuilder)
    position_bounds: tuple[float, float] = (-1.0, 1.0)

    # Populated during act() so the training loop knows what happened at this step
    _last_pid_input: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_action: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        expected = PIDNetInputBuilder.input_size(len(self.normalizer))
        if self.pid_net.input_size != expected:
            raise ValueError(
                f"pid_net.input_size is {self.pid_net.input_size}, but {len(self.normalizer)} "
                f"features need {expected}; build PIDNet with input_size={expected}"
            )

    @property
    def required_fields(self) -> set[str]:
        # realized_vol is read directly by the actor/critic in the training loop
        return set(PID_LAW_FIELDS) | set(self.normalizer.names) | {"realized_vol"}

    def act(self, state: pd.Series) -> float:
        """Returns a target position, running PIDNet in inference mode.

        Uses raw e_p/e_i/e_d for the PID formula and normalized features only
        as input to the network: the linear PID law needs its inputs in
        original units to give a meaningful position, while the network
        learns better on normalized inputs.
        """
        require_columns(state, self.required_fields, "A2CPID.act")

        pid_input = self.input_builder.build(self.normalizer.transform(state))

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

        return action

    def record_observation(self, position: float) -> None:
        """Updates the position history after env.step has been executed."""
        self.input_builder.record(position=position)

    def build_training_inputs(self) -> tuple[torch.Tensor, float]:
        """Returns the (pid_input, action) captured during the last act() call.

        Meant for the training loop, which needs these to compute losses
        against the observed next state and reward.
        """
        if self._last_pid_input is None or self._last_action is None:
            raise RuntimeError("build_training_inputs called before any act()")
        return self._last_pid_input, self._last_action
