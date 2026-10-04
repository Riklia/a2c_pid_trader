import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


DIRECTION_MODES = ("long_only", "trend_long_flat", "trend_long_short")


@dataclass(frozen=True)
class PIDErrors:
    """Error signals of the log-space PID at one step, all dimensionless.

        e = log(target_vol / vol_forecast)       market vol vs setpoint
        e_int = sum of log(target_vol / portfolio_vol) over past bars,
                anti-windup protected; closes the loop on what the
                portfolio actually realized
        e_d = one-bar change of e
    """
    e: float
    e_int: float
    e_d: float


@dataclass(frozen=True)
class StepContext:
    """Everything the law needs to turn gains into a position at one step."""
    direction: float  # -1, 0 or +1
    errors: PIDErrors
    feedback_error: float  # log(target_vol / portfolio_vol), 0 while warming up


def _clip(x, low: float, high: float):
    """Clips floats and torch tensors alike, keeping the gradient for tensors."""
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, low, high)
    return min(max(x, low), high)


@dataclass
class VolTargetLaw:
    """Log-space PID volatility targeting, split into direction and size.

        position = direction * clip(exp(kp * e + ki * e_int + kd * e_d), 0, max_position)

    With kp = 1 and ki = kd = 0 this is exactly textbook vol targeting,
    exposure = target_vol / vol_forecast. The gains then have a direct
    reading: kp > 1 de-risks harder in turbulent markets and levers up more
    in calm ones, ki corrects persistent gaps between realized portfolio vol
    and the target (the closed-loop part, since it depends on past actions),
    kd reacts to vol trending up or down.

    The law is stateful: it remembers its last position and the returns
    that position earned. Call begin_step exactly once per bar, then commit
    with the gains actually used. position() is pure and accepts tensor
    gains, so training can differentiate through it.

    Attributes:
        target_vol: annualized portfolio vol to track.
        max_position: cap on absolute exposure, as a multiple of equity.
        integral_limit: cap on |e_int|, in bars of unit log error.
        port_vol_window: bars of strategy returns used to measure portfolio vol.
            The integral is frozen until this many returns are observed.
        bars_per_year: annualization factor for portfolio vol.
        direction_mode: "long_only", "trend_long_flat" (long when trend > 0,
            flat otherwise) or "trend_long_short" (sign of trend).
        trend_column: state column the trend modes read the signal from.
        direction_hold: bars a direction decision is kept before the signal
            is read again. A daily re-evaluation (24) stops a noisy signal from
            flipping the position, and paying fees, every hour.
    """
    target_vol: float
    max_position: float = 1.0
    integral_limit: float = 100.0
    port_vol_window: int = 168
    bars_per_year: float = 8760
    direction_mode: str = "long_only"
    trend_column: str = "trend"
    direction_hold: int = 1

    _returns: deque = field(init=False, repr=False)
    _last_position: float = field(init=False, repr=False)
    _e_int: float = field(init=False, repr=False)
    _e_prev: float | None = field(init=False, repr=False)
    _direction: float = field(init=False, repr=False)
    _direction_age: int = field(init=False, repr=False)

    def __post_init__(self):
        if self.target_vol <= 0:
            raise ValueError("target_vol must be positive")
        if self.direction_mode not in DIRECTION_MODES:
            raise ValueError(f"unknown direction_mode: {self.direction_mode}")
        if self.port_vol_window < 2:
            raise ValueError("port_vol_window must be at least 2")
        if self.direction_hold < 1:
            raise ValueError("direction_hold must be at least 1")
        self.reset()

    @property
    def required_fields(self) -> set[str]:
        fields = {"log_return", "vol_forecast"}
        if self.direction_mode != "long_only":
            fields.add(self.trend_column)
        return fields

    def reset(self) -> None:
        """Forgets position, return history and the integral, as at episode start."""
        self._returns = deque(maxlen=self.port_vol_window)
        self._last_position = 0.0
        self._e_int = 0.0
        self._e_prev = None
        self._direction = 0.0
        self._direction_age = self.direction_hold  # forces a fresh read on the first bar

    def portfolio_vol(self) -> float | None:
        """Annualized vol of recent strategy returns, None until the window is full."""
        if len(self._returns) < self.port_vol_window:
            return None
        arr = np.fromiter(self._returns, dtype=np.float64, count=len(self._returns))
        return float(arr.std(ddof=1) * math.sqrt(self.bars_per_year))

    def begin_step(self, state: pd.Series) -> StepContext:
        """Books the return the last position earned on this bar, returns the step context."""
        log_return = float(state["log_return"])
        if math.isfinite(log_return):
            self._returns.append(self._last_position * math.expm1(log_return))

        e = math.log(self.target_vol / float(state["vol_forecast"]))
        e_d = 0.0 if self._e_prev is None else e - self._e_prev
        self._e_prev = e

        port_vol = self.portfolio_vol()
        feedback_error = 0.0 if not port_vol else math.log(self.target_vol / port_vol)

        return StepContext(
            direction=self._next_direction(state),
            errors=PIDErrors(e=e, e_int=self._e_int, e_d=e_d),
            feedback_error=feedback_error,
        )

    def log_exposure(self, ctx: StepContext, kp, ki, kd):
        """PID output before exp and clipping, float or tensor depending on gains."""
        err = ctx.errors
        return kp * err.e + ki * err.e_int + kd * err.e_d

    def position(self, ctx: StepContext, kp, ki, kd):
        """Pure: the position these gains produce. Differentiable for tensor gains."""
        log_exp = self.log_exposure(ctx, kp, ki, kd)
        exposure = torch.exp(log_exp) if isinstance(log_exp, torch.Tensor) else math.exp(log_exp)
        return ctx.direction * _clip(exposure, 0.0, self.max_position)

    def commit(self, ctx: StepContext, kp: float, ki: float, kd: float) -> float:
        """Returns the position for these gains and advances the integral and position state.

        Anti-windup by conditional integration: when exposure is pinned at
        max_position, a positive feedback error (portfolio vol below target)
        is not accumulated. Otherwise e_int would keep growing during long
        calm stretches and overshoot once vol picks up.
        """
        position = float(self.position(ctx, kp, ki, kd))

        fb = ctx.feedback_error
        saturated = abs(position) >= self.max_position
        if ctx.direction != 0 and not (saturated and fb > 0):
            self._e_int = _clip(self._e_int + fb, -self.integral_limit, self.integral_limit)

        self._last_position = position
        return position

    def _next_direction(self, state: pd.Series) -> float:
        """Reads the signal every direction_hold bars, otherwise keeps the last decision."""
        if self._direction_age >= self.direction_hold:
            self._direction = self._read_direction(state)
            self._direction_age = 0
        self._direction_age += 1
        return self._direction

    def _read_direction(self, state: pd.Series) -> float:
        if self.direction_mode == "long_only":
            return 1.0
        trend = float(state[self.trend_column])
        if self.direction_mode == "trend_long_flat":
            return 1.0 if trend > 0 else 0.0
        return float(np.sign(trend))
