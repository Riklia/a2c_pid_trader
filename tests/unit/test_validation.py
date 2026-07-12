import pandas as pd
import pytest

from data.validation import require_columns, require_sorted_unique_datetime_index


def test_require_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    require_columns(df, {"a", "b"}, "test")  # should not raise


def test_require_columns_raises_with_context_and_missing_names():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"my_ctx: missing required fields: \['b', 'c'\]"):
        require_columns(df, {"a", "b", "c"}, "my_ctx")


def test_require_columns_works_on_series():
    s = pd.Series({"a": 1, "b": 2})
    require_columns(s, {"a"}, "test")  # should not raise
    with pytest.raises(ValueError, match="missing required fields"):
        require_columns(s, {"a", "missing"}, "test")


def test_datetime_index_validator_passes_on_good_index():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="1h"))
    require_sorted_unique_datetime_index(df, "test")


def test_datetime_index_validator_rejects_non_datetime_index():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
    with pytest.raises(ValueError, match="must be a DatetimeIndex"):
        require_sorted_unique_datetime_index(df, "test")


def test_datetime_index_validator_rejects_unsorted():
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    with pytest.raises(ValueError, match="must be sorted ascending"):
        require_sorted_unique_datetime_index(df, "test")


def test_datetime_index_validator_rejects_duplicates():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        require_sorted_unique_datetime_index(df, "test")