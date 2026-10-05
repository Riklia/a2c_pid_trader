import pandas as pd
import pytest

from backtest.walk_forward import make_walk_forward_folds


def _fake_data(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"x": range(n)}, index=idx)


def test_produces_expected_number_of_folds():
    data = _fake_data(n=100)
    folds = make_walk_forward_folds(data, train_size=50, test_size=10)
    # 100 bars, need 60 per fold, step=10 -> starts at 0,10,20,30,40 -> 5 folds
    assert len(folds) == 5


def test_folds_dont_overlap_between_train_and_test():
    data = _fake_data(n=100)
    folds = make_walk_forward_folds(data, train_size=50, test_size=10)
    for fold in folds:
        assert fold.train_data.index[-1] < fold.test_data.index[0]


def test_test_windows_are_contiguous_by_default():
    data = _fake_data(n=100)
    folds = make_walk_forward_folds(data, train_size=50, test_size=10)
    for prev, curr in zip(folds, folds[1:]):
        assert prev.test_data.index[-1] < curr.test_data.index[0]
        # step_size=test_size=10, so gap between consecutive test starts is exactly 10 hours
        assert (curr.test_start - prev.test_start) == pd.Timedelta("10h")


def test_custom_step_size_creates_overlapping_windows():
    data = _fake_data(n=100)
    folds = make_walk_forward_folds(data, train_size=50, test_size=10, step_size=5)
    assert len(folds) == 9  # starts at 0,5,10,...,40


def test_no_folds_when_data_too_short():
    data = _fake_data(n=30)
    folds = make_walk_forward_folds(data, train_size=50, test_size=10)
    assert folds == []


def test_fold_metadata_properties():
    data = _fake_data(n=100)
    fold = make_walk_forward_folds(data, train_size=50, test_size=10)[0]
    assert fold.train_start == data.index[0]
    assert fold.train_end == data.index[49]
    assert fold.test_start == data.index[50]
    assert fold.test_end == data.index[59]


def test_expanding_window_trains_on_all_history_with_the_same_test_windows():
    data = _fake_data(n=100)
    rolling = make_walk_forward_folds(data, train_size=50, test_size=10)
    expanding = make_walk_forward_folds(data, train_size=50, test_size=10, window="expanding")
    assert [f.test_data.index[0] for f in expanding] == [f.test_data.index[0] for f in rolling]
    for fold in expanding:
        assert fold.train_data.index[0] == data.index[0]
        assert fold.train_data.index[-1] < fold.test_data.index[0]
    assert len(expanding[-1].train_data) == 90


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError, match="unknown window"):
        make_walk_forward_folds(_fake_data(), train_size=50, test_size=10, window="sliding")
