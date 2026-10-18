from dataclasses import dataclass, field

import pandas as pd
import torch

from control.static_pid import StaticPID
from control.vol_target import StepContext, VolTargetLaw
from data.model_features import DEFAULT_FEATURES, FeatureNormalizer
from data.validation import require_columns
from models.pid_net import PIDNet, PIDNetInputBuilder


@dataclass
class A2CPID:
    """Self-tuning vol-targeting PID: static gains plus network-driven dynamic gains.

    Uses the same control law as StaticPID (see VolTargetLaw), only the gains
    differ:
        kp = kp_static + kp_dyn, same for ki and kd
        [kp_dyn, ki_dyn, kd_dyn] = PIDNet([u(k-1), u(k-2), law errors, model features])

    The network sees whatever features `normalizer` is configured with (see
    data.model_features). With a freshly initialized PIDNet the dynamic gains
    are near zero, so the controller starts out as its static PID.

    Attributes:
        static_pid: provides the law and the static gains.
        pid_net: network producing dynamic gains added to the static ones. Its
            input_size must match the normalizer's feature count.
        normalizer: selects and normalizes the model features.
        input_builder: keeps the 2-step position history for the pid_net input.
        gain_hold: bars a network gain prediction is held before re-evaluating
            PIDNet. Setting this to 24 (matching direction_hold and update_every)
            eliminates hourly fee chatter while keeping the PID law active every bar.
    """

    static_pid: StaticPID
    pid_net: PIDNet
    normalizer: FeatureNormalizer = field(default_factory=lambda: FeatureNormalizer(DEFAULT_FEATURES))
    input_builder: PIDNetInputBuilder = field(default_factory=PIDNetInputBuilder)
    gain_hold: int = 1

    # Populated during act() so training can rebuild the action with gradient
    _last_pid_input: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_ctx: StepContext | None = field(default=None, init=False, repr=False)
    _held_dyn: tuple[float, ...] | None = field(default=None, init=False, repr=False)
    _gain_age: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.gain_hold < 1:
            raise ValueError("gain_hold must be at least 1")
        expected = PIDNetInputBuilder.input_size(len(self.normalizer))
        if self.pid_net.input_size != expected:
            raise ValueError(
                f"pid_net.input_size is {self.pid_net.input_size}, but {len(self.normalizer)} "
                f"features need {expected}; build PIDNet with input_size={expected}"
            )
        self._gain_age = self.gain_hold

    @property
    def law(self) -> VolTargetLaw:
        return self.static_pid.law

    @property
    def required_fields(self) -> set[str]:
        return self.law.required_fields | set(self.normalizer.names)

    def act(self, state: pd.Series) -> float:
        """Returns a target position, running PIDNet in inference mode."""
        require_columns(state, self.required_fields, "A2CPID.act")

        ctx = self.law.begin_step(state)
        norm_features = self.normalizer.transform(state)
        if self._gain_age >= self.gain_hold or self._last_pid_input is None or self._held_dyn is None:
            pid_input = self.input_builder.build(ctx.errors, self.law.integral_limit, norm_features)
            with torch.no_grad():
                dyn = self.pid_net(pid_input)
            self._held_dyn = tuple(float(g) for g in dyn)
            self._last_pid_input = pid_input
            self._gain_age = 0
        self._gain_age += 1

        kp_d, ki_d, kd_d = self._held_dyn[0], self._held_dyn[1], self._held_dyn[2]
        bias_d = self._held_dyn[3] if len(self._held_dyn) == 4 else 0.0

        action = self.law.commit(
            ctx,
            self.static_pid.kp + kp_d,
            self.static_pid.ki + ki_d,
            self.static_pid.kd + kd_d,
            bias=bias_d,
        )
        self.input_builder.record(action)

        self._last_ctx = ctx
        return action

    def differentiable_action(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Recomputes the last act() output with gradient flowing into pid_net.

        Returns (position, dynamic_gains). Meant for the training loop, which
        backpropagates a loss through the position. Must be called before
        pid_net's weights change, otherwise it will not match the executed action.
        """
        if self._last_pid_input is None or self._last_ctx is None:
            raise RuntimeError("differentiable_action called before any act()")
        dyn = self.pid_net(self._last_pid_input)
        bias_d = dyn[3] if dyn.shape[-1] == 4 else 0.0
        position = self.law.position(
            self._last_ctx,
            self.static_pid.kp + dyn[0],
            self.static_pid.ki + dyn[1],
            self.static_pid.kd + dyn[2],
            bias=bias_d,
        )
        return position, dyn

    def reset(self) -> None:
        """Starts a new episode. Network weights and normalization statistics are kept."""
        self.law.reset()
        self.input_builder.reset()
        self._last_pid_input = None
        self._last_ctx = None
        self._held_dyn = None
        self._gain_age = self.gain_hold
