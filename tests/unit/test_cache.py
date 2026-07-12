# tests/unit/test_cache.py
import pandas as pd
import pytest

from data.cache import cache_path_for, load_from_cache, save_to_cache


@pytest.fixture
def sample_df():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [True, False, True]}, index=idx)


def test_cache_path_is_deterministic():
    params = {"symbol": "BTCUSDT", "freq": "1h"}
    assert cache_path_for(params) == cache_path_for(params)

def test_cache_path_changes_with_params():
    p1 = cache_path_for({"symbol": "BTCUSDT", "freq": "1h"})
    p2 = cache_path_for({"symbol": "BTCUSDT", "freq": "4h"})
    assert p1 != p2

def test_cache_path_invariant_to_key_order():
    p1 = cache_path_for({"symbol": "BTCUSDT", "freq": "1h"})
    p2 = cache_path_for({"freq": "1h", "symbol": "BTCUSDT"})
    assert p1 == p2

def test_load_returns_none_when_missing(tmp_path):
    result = load_from_cache({"symbol": "MISSING"}, cache_dir=tmp_path)
    assert result is None

def test_save_and_load_roundtrip(tmp_path, sample_df):
    params = {"symbol": "BTCUSDT", "freq": "1h"}
    save_to_cache(sample_df, params, cache_dir=tmp_path)
    loaded = load_from_cache(params, cache_dir=tmp_path)

    pd.testing.assert_frame_equal(loaded, sample_df, check_freq=False)

def test_save_creates_cache_dir_if_missing(tmp_path, sample_df):
    nested = tmp_path / "does" / "not" / "exist"
    save_to_cache(sample_df, {"x": 1}, cache_dir=nested)
    assert (nested).exists()
