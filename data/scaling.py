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
        self._buffers: dict[str, np.ndarray] = {
            name: np.empty(window_size, dtype=np.float64) for name in self.feature_names
        }
        self._counts: dict[str, int] = {name: 0 for name in self.feature_names}
        self._pos: dict[str, int] = {name: 0 for name in self.feature_names}

    def transform_value(self, name: str, value: float) -> float:
        """Updates rolling stats for `name` with `value` and returns its z-score."""
        val = float(value)
        buf = self._buffers[name]
        pos = self._pos[name]
        buf[pos] = val
        self._pos[name] = (pos + 1) % self.window_size
        count = self._counts[name]
        if count < self.window_size:
            count += 1
            self._counts[name] = count

        if count < self.warmup_size:
            return 0.0

        view = buf if count == self.window_size else buf[:count]
        mean = float(view.mean())
        std = float(view.std(ddof=1))
        return (val - mean) / (std + self.eps)

    def transform(self, sample: pd.Series) -> pd.Series:
        """Updates rolling stats with `sample` and returns a standardized copy."""
        result = sample.copy()
        for name in self.feature_names:
            result[name] = self.transform_value(name, float(sample[name]))
        return result

    def reset(self) -> None:
        """Clears all rolling windows."""
        for name in self.feature_names:
            self._counts[name] = 0
            self._pos[name] = 0