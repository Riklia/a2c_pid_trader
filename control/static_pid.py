from dataclasses import dataclass

import pandas as pd

from control.vol_target import VolTargetLaw
from data.validation import require_columns


@dataclass
class StaticPID:
    """Fixed-gain vol-targeting PID. See VolTargetLaw for the control law.

    With the default gains (kp=1, ki=kd=0) this is plain vol targeting,
    position = target_vol / vol_forecast, capped.

    Attributes:
        law: stateful control law shared with A2CPID, owns target, caps and
            the feedback loop state.
        kp: proportional gain on log(target_vol / vol_forecast).
        ki: integral gain, per bar of accumulated log portfolio-vol error.
        kd: derivative gain on the one-bar change of the proportional error.
    """
    law: VolTargetLaw
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0

    def act(self, state: pd.Series) -> float:
        require_columns(state, self.law.required_fields, "StaticPID.act")
        ctx = self.law.begin_step(state)
        return self.law.commit(ctx, self.kp, self.ki, self.kd)

    def reset(self) -> None:
        self.law.reset()
