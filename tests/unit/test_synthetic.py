import numpy as np
import pytest

from data.synthetic import lagged_vol_forecast, make_market, regime_switching_vol


def test_regime_switching_vol_only_uses_given_levels_and_always_switches():
    vol = regime_switching_vol(5000, (0.3, 0.6, 1.2), mean_duration=200, seed=1)
    assert set(np.unique(vol)) <= {0.3, 0.6, 1.2}
    changes = np.flatnonzero(np.diff(vol) != 0)
    assert len(changes) > 5
    # mean regime length is roughly the requested one
    assert 100 < len(vol) / (len(changes) + 1) < 400


def test_lagged_forecast_is_causal():
    lr = make_market(n=600, seed=2)["log_return"].to_numpy()
    full = lagged_vol_forecast(lr, 168)
    cut = lagged_vol_forecast(lr[:400], 168)
    np.testing.assert_allclose(full[:400], cut)


def test_lagged_forecast_lags_a_vol_jump():
    vol = np.r_[np.full(1000, 0.3), np.full(1000, 1.2)]
    lr = make_market(n=2000, vol=vol, seed=3)["log_return"].to_numpy()
    forecast = lagged_vol_forecast(lr, 168)
    assert forecast[1010] < 0.6  # still mostly the old regime right after the jump
    assert forecast[1500] == pytest.approx(1.2, rel=0.2)  # caught up a window later
