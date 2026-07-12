from typing import Protocol

import pandas as pd


class Controller(Protocol):
    """Anything that turns a state row into a target position in [-1, 1]."""

    def act(self, state: pd.Series) -> float:
        ...
