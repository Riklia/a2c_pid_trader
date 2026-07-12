import numpy as np
import pandas as pd
import pytest

from data import features


@pytest.fixture
def quarter_hourly_ohlcv():
    """8 bars at 15min resolution, two full 1h buckets, for resample tests."""
    idx = pd.to_datetime([
        "2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30", "2024-01-01 00:45",
        "2024-01-01 01:00", "2024-01-01 01:15", "2024-01-01 01:30", "2024-01-01 01:45",
    ])
    return pd.DataFrame({
        "open":   [100, 101, 102, 103, 110, 111, 112, 113],
        "high":   [105, 104, 107, 106, 115, 114, 117, 116],
        "low":    [95,  97,  94,  98,  105, 107, 104, 108],
        "close":  [101, 102, 103, 104, 111, 112, 113, 114],
        "volume": [1,   1,   1,   1,   2,   2,   2,   2],
        "trades": [10,  10,  10,  10,  20,  20,  20,  20],
    }, index=idx)


@pytest.fixture
def hourly_ohlcv():
    """9 flat-price hourly bars, used where only timing (not price) matters."""
    idx = pd.date_range("2024-01-01 00:00", periods=9, freq="1h")
    return pd.DataFrame({
        "open": [100.0] * 9, "high": [100.0] * 9, "low": [100.0] * 9, "close": [100.0] * 9,
        "volume": [1.0] * 9, "trades": [10] * 9,
    }, index=idx)


@pytest.fixture
def varying_hourly_ohlcv():
    """9 hourly bars with non-trivial price moves, for tests that need
    nonzero returns and volatility to catch calculation/alignment errors."""
    idx = pd.date_range("2024-01-01 00:00", periods=9, freq="1h")
    close = [100.0, 102.0, 99.0, 103.0, 97.0, 104.0, 96.0, 105.0, 95.0]
    return pd.DataFrame({
        "open": close, "high": [c + 1 for c in close], "low": [c - 1 for c in close],
        "close": close, "volume": [1.0] * 9, "trades": [10] * 9,
    }, index=idx)


@pytest.fixture
def eight_hour_funding():
    """Funding updates at hour 0 and hour 8, matching hourly_ohlcv's range."""
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"])
    return pd.DataFrame({"mark_price": [100.0, 105.0], "funding": [0.0001, 0.0002]}, index=idx)


@pytest.fixture
def constant_returns():
    """4 hourly log returns, values chosen so std is easy to check by hand."""
    idx = pd.date_range("2024-01-01 00:00", periods=4, freq="1h")
    return pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)


@pytest.fixture
def constant_error_vol():
    """Realized vol fixed at 0.5 against a target of 1.5 -> constant e_p of 1.0."""
    idx = pd.date_range("2024-01-01", periods=6, freq="1h")
    return pd.Series([0.5] * 6, index=idx)


def test_resample_ohlcv_aggregates_full_buckets_correctly(quarter_hourly_ohlcv):
    result = features.resample_ohlcv(quarter_hourly_ohlcv, freq="1h")

    expected = pd.DataFrame({
        "open": [100.0, 110.0], "high": [107.0, 117.0], "low": [94.0, 104.0],
        "close": [104.0, 114.0], "volume": [4.0, 8.0], "trades": [40, 80],
    }, index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00"]))

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_freq=False)


def test_resample_ohlcv_drops_empty_buckets(quarter_hourly_ohlcv):
    gapped = quarter_hourly_ohlcv.iloc[:4]
    extra = quarter_hourly_ohlcv.iloc[4:].copy()
    extra.index = extra.index + pd.Timedelta("1h")  # pushes data to hour 2, hour 1 stays empty
    combined = pd.concat([gapped, extra])

    result = features.resample_ohlcv(combined, freq="1h")

    assert pd.Timestamp("2024-01-01 01:00") not in result.index
    assert pd.Timestamp("2024-01-01 02:00") in result.index


def test_resample_ohlcv_keeps_bucket_with_single_observation(quarter_hourly_ohlcv):
    """A bucket built from a single quarter-hourly bar is intentionally kept:
    a partially formed final candle is a valid data point, not something to drop."""
    partial = pd.concat([quarter_hourly_ohlcv.iloc[:4], quarter_hourly_ohlcv.iloc[4:5]])

    result = features.resample_ohlcv(partial, freq="1h")

    expected_row = pd.Series(
        {"open": 110.0, "high": 115.0, "low": 105.0, "close": 111.0, "volume": 2.0, "trades": 20},
        name=pd.Timestamp("2024-01-01 01:00:00"),
    )
    pd.testing.assert_series_equal(result.loc["2024-01-01 01:00"], expected_row, check_dtype=False)


def test_funding_is_forward_filled_between_updates(hourly_ohlcv, eight_hour_funding):
    result = features.align_funding_to_ohlcv(hourly_ohlcv, eight_hour_funding, freq="1h")

    expected_mark_price = pd.Series(
        [100.0] * 8 + [105.0], index=hourly_ohlcv.index, name="mark_price",
    )
    pd.testing.assert_series_equal(result["mark_price"], expected_mark_price, check_freq=False)


def test_funding_does_not_leak_from_the_future(hourly_ohlcv, eight_hour_funding):
    result = features.align_funding_to_ohlcv(hourly_ohlcv, eight_hour_funding, freq="1h")

    assert result.loc["2024-01-01 07:00", "mark_price"] == 100.0
    assert result.loc["2024-01-01 08:00", "mark_price"] == 105.0


def test_funding_off_grid_update_is_captured_within_its_bar(hourly_ohlcv):
    """An update at 08:30 lands in the bar [08:00, 09:00), which only
    closes at 09:00 - so it's known by the time that bar's close is known."""
    funding = pd.DataFrame(
        {"mark_price": [100.0, 108.0], "funding": [0.0001, 0.0005]},
        index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:30"]),
    )

    result = features.align_funding_to_ohlcv(hourly_ohlcv, funding, freq="1h")

    assert result.loc["2024-01-01 07:00", "mark_price"] == 100.0
    assert result.loc["2024-01-01 08:00", "mark_price"] == 108.0


def test_funding_is_nan_before_first_observation(hourly_ohlcv):
    funding = pd.DataFrame(
        {"mark_price": [105.0], "funding": [0.0002]},
        index=pd.to_datetime(["2024-01-01 03:00"]),
    )

    result = features.align_funding_to_ohlcv(hourly_ohlcv, funding, freq="1h")

    assert result.loc["2024-01-01 00:00":"2024-01-01 02:00", "mark_price"].isna().all()
    assert result.loc["2024-01-01 03:00", "mark_price"] == 105.0


def test_realized_vol_matches_hand_calculated_std(constant_returns):
    vol = features.compute_realized_vol(constant_returns, freq="1h", window="3h", annualize=False)
    # std of [0.01, 0.02, 0.03] with ddof=1: mean=0.02, var=0.0001, std=0.01
    assert vol.iloc[2] == pytest.approx(0.01)


def test_realized_vol_first_valid_value_appears_after_full_window(constant_returns):
    vol = features.compute_realized_vol(constant_returns, freq="1h", window="3h", annualize=False)
    assert vol.iloc[:2].isna().all()
    assert not pd.isna(vol.iloc[2])


def test_realized_vol_annualization_factor_is_exact(constant_returns):
    raw = features.compute_realized_vol(constant_returns, freq="1h", window="3h", annualize=False)
    annualized = features.compute_realized_vol(constant_returns, freq="1h", window="3h", annualize=True)
    expected = 0.01 * np.sqrt(24 * 365)
    assert annualized.iloc[2] == pytest.approx(expected)


def test_realized_vol_window_shorter_than_bar_raises():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    returns = pd.Series([0.01, 0.02, 0.03], index=idx)
    with pytest.raises(ValueError, match="shorter than one bar"):
        features.compute_realized_vol(returns, freq="1h", window="30min")


def test_proportional_error_is_target_minus_realized(constant_error_vol):
    result = features.compute_pid_errors(constant_error_vol, target_vol=1.5)
    assert (result["e_p"] == 1.0).all()


def test_rolling_integral_caps_at_window_sum(constant_error_vol):
    result = features.compute_pid_errors(
        constant_error_vol, target_vol=1.5, integral_mode="rolling", integral_window="3h"
    )
    assert list(result["e_i"]) == pytest.approx([1, 2, 3, 3, 3, 3])


def test_expanding_integral_keeps_accumulating(constant_error_vol):
    result = features.compute_pid_errors(constant_error_vol, target_vol=1.5, integral_mode="expanding")
    assert list(result["e_i"]) == pytest.approx([1, 2, 3, 4, 5, 6])


def test_derivative_is_diff_of_proportional_error():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    vol = pd.Series([0.5, 0.6, 0.5, 0.5], index=idx)
    result = features.compute_pid_errors(vol, target_vol=1.0)

    assert pd.isna(result["e_d"].iloc[0])
    assert result["e_d"].iloc[1] == pytest.approx(-0.1)
    assert result["e_d"].iloc[2] == pytest.approx(0.1)


def test_unknown_integral_mode_raises(constant_error_vol):
    with pytest.raises(ValueError, match="unknown integral_mode"):
        features.compute_pid_errors(constant_error_vol, target_vol=1.5, integral_mode="nonsense")


def test_build_feature_table_has_expected_columns(varying_hourly_ohlcv, eight_hour_funding):
    result = features.build_feature_table(
        varying_hourly_ohlcv, eight_hour_funding, target_vol=0.1, freq="1h",
        vol_window="3h", integral_window="3h",
    )
    assert list(result.columns) == [
        "open", "high", "low", "close", "volume", "trades",
        "mark_price", "funding", "is_funding_settlement", "log_return",
        "realized_vol", "e_p", "e_i", "e_d",
    ]


def test_build_feature_table_drops_warmup_rows(varying_hourly_ohlcv, eight_hour_funding):
    result = features.build_feature_table(
        varying_hourly_ohlcv, eight_hour_funding, target_vol=0.1, freq="1h",
        vol_window="3h", integral_window="3h",
    )
    # row0: no return yet. rows1-2: return exists but vol window (3 bars) not full.
    # row3: vol/e_p valid but e_d has no prior e_p to diff against. row4: first complete row.
    assert result.index[0] == pd.Timestamp("2024-01-01 04:00")
    assert result.isna().sum().sum() == 0


def test_build_feature_table_has_no_look_ahead(varying_hourly_ohlcv, eight_hour_funding):
    cutoff = pd.Timestamp("2024-01-01 06:00")

    full = features.build_feature_table(
        varying_hourly_ohlcv, eight_hour_funding, target_vol=0.1, freq="1h",
        vol_window="3h", integral_window="3h",
    )
    truncated = features.build_feature_table(
        varying_hourly_ohlcv[varying_hourly_ohlcv.index <= cutoff],
        eight_hour_funding[eight_hour_funding.index <= cutoff],
        target_vol=0.1, freq="1h", vol_window="3h", integral_window="3h",
    )

    shared_index = truncated.index
    pd.testing.assert_frame_equal(full.loc[shared_index], truncated)

def test_funding_settlement_flag_marks_only_real_observations(hourly_ohlcv, eight_hour_funding):
    result = features.align_funding_to_ohlcv(hourly_ohlcv, eight_hour_funding, freq="1h")

    expected = pd.Series(
        [True] + [False] * 7 + [True], index=hourly_ohlcv.index, name="is_funding_settlement",
    )
    pd.testing.assert_series_equal(result["is_funding_settlement"], expected, check_freq=False)