from typing import Protocol, runtime_checkable

import pandas as pd

@runtime_checkable
class Controller(Protocol):
    """Anything that turns a state row into a target position in [-1, 1]."""

    def act(self, state: pd.Series) -> float:
        ...
