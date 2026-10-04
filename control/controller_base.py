from typing import Protocol, runtime_checkable

import pandas as pd

@runtime_checkable
class Controller(Protocol):
    """Anything that turns a state row into a target position.

    Controllers may keep state across bars (e.g. their own past positions),
    reset() returns them to the start-of-episode state.
    """

    def act(self, state: pd.Series) -> float:
        ...

    def reset(self) -> None:
        ...
