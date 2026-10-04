from dataclasses import dataclass

import pandas as pd


@dataclass
class BuyAndHold:
    """Holds a constant long position. The no-skill benchmark every strategy should face.

    A long-biased controller that only beats zero in up markets is just
    harvesting the asset's drift; comparing against this makes that visible.
    """
    position: float = 1.0

    def act(self, state: pd.Series) -> float:
        return self.position
