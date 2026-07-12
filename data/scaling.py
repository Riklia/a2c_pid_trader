from collections import deque

import numpy as np
import pandas as pd


class RollingZScoreScaler:
    """Standardizes each feature by its rolling mean and std.

    Maintains a fixed-length window of recent observations per feature and
    transforms new samples using running statistics from that window. Works
    online: no fit step needed, statistics update on every call to transform.

    Before the window has warmup_size samples, transform returns zeros so
    unstable early estimates never reach the network.

    Args:
        feature_names: names of the features to standardize. Other columns
            in the input are passed through unchanged.
        window_size: how many past samples to keep for the running stats.
        warmup_size: minimum samples required before real z-scores are
            returned. Earlier calls return zeros.
        eps: added to std before division to avoid divide-by-zero on
            constant features.
    """

    def __init__(
        self,
        feature_names: list[str],
        window_size: int = 500,
        warmup_size: int = 50,
        eps: float = 1e-8,
    ):
        if warmup_size < 2:
            raise ValueError("warmup_size must be at least 2 to compute std")
        if warmup_size > window_size:
            raise ValueError("warmup_size cannot exceed window_size")

        self.feature_names = list(feature_names)
        self.window_size = window_size
        self.warmup_size = warmup_size
        self.eps = eps
        self._windows: dict[str, deque[float]] = {
            name: deque(maxlen=window_size) for name in self.feature_names
        }

    def transform(self, sample: pd.Series) -> pd.Series:
        """Updates rolling stats with `sample` and returns a standardized copy."""
        result = sample.copy()
        for name in self.feature_names:
            value = float(sample[name])
            window = self._windows[name]
            window.append(value)

            if len(window) < self.warmup_size:
                result[name] = 0.0
                continue

            arr = np.fromiter(window, dtype=np.float64, count=len(window))
            mean = arr.mean()
            std = arr.std(ddof=1)
            result[name] = (value - mean) / (std + self.eps)
        return result

    def reset(self) -> None:
        """Clears all rolling windows."""
        for window in self._windows.values():
            window.clear()