import numpy as np
import pandas as pd
import pytest

from data.scaling import RollingZScoreScaler


def test_returns_zero_during_warmup():
    scaler = RollingZScoreScaler(feature_names=["x"], window_size=100, warmup_size=5)
    for i in range(4):
        out = scaler.transform(pd.Series({"x": float(i)}))
        assert out["x"] == 0.0


def test_returns_real_zscore_after_warmup():
    scaler = RollingZScoreScaler(feature_names=["x"], window_size=100, warmup_size=3)
    for value in [1.0, 2.0, 3.0]:
        out = scaler.transform(pd.Series({"x": value}))
    # window is now [1, 2, 3], mean=2, std(ddof=1)=1, last value=3
    # z = (3 - 2) / (1 + eps) ~= 1
    assert out["x"] == pytest.approx(1.0, abs=1e-6)


def test_constant_feature_produces_finite_output():
    """std=0 must not blow up thanks to eps."""
    scaler = RollingZScoreScaler(feature_names=["x"], window_size=100, warmup_size=3)
    for _ in range(10):
        out = scaler.transform(pd.Series({"x": 5.0}))
    assert np.isfinite(out["x"])
    assert out["x"] == pytest.approx(0.0, abs=1e-6)


def test_non_feature_columns_are_passed_through():
    scaler = RollingZScoreScaler(feature_names=["x"], window_size=100, warmup_size=3)
    out = scaler.transform(pd.Series({"x": 1.0, "keep_as_is": "hello"}))
    assert out["keep_as_is"] == "hello"


def test_reset_clears_window():
    scaler = RollingZScoreScaler(feature_names=["x"], window_size=100, warmup_size=3)
    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        scaler.transform(pd.Series({"x": value}))
    scaler.reset()
    out = scaler.transform(pd.Series({"x": 100.0}))
    # window is empty again, we should be back in warmup
    assert out["x"] == 0.0


def test_rolling_window_forgets_old_samples():
    scaler = RollingZScoreScaler(feature_names=["x"], window_size=5, warmup_size=3)
    scaler.transform(pd.Series({"x": 100.0}))  # outlier that should be forgotten
    for value in [1.0, 1.0, 1.0, 1.0, 1.0]:  # fills the 5-slot window with 1s
        scaler.transform(pd.Series({"x": value}))
    # window is now [1, 1, 1, 1, 1], outlier is gone
    # std of [1,1,1,1,1] = 0 -> z-score of new 1.0 should be ~0
    out = scaler.transform(pd.Series({"x": 1.0}))
    assert out["x"] == pytest.approx(0.0, abs=1e-6)


def test_warmup_cannot_exceed_window():
    with pytest.raises(ValueError, match="cannot exceed"):
        RollingZScoreScaler(feature_names=["x"], window_size=10, warmup_size=20)


def test_warmup_below_two_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        RollingZScoreScaler(feature_names=["x"], window_size=100, warmup_size=1)
