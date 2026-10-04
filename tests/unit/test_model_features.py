import numpy as np
import pandas as pd
import pytest

from data.model_features import (
    COLUMN_DESCRIPTIONS, FeatureNormalizer, FeatureSpec,
    attach_cross_asset_features, describe_features, other_symbols, parse_feature_specs,
    with_availability_flags,
)


def _table(values, start="2024-01-01 00:00", column="realized_vol"):
    idx = pd.date_range(start, periods=len(values), freq="1h")
    return pd.DataFrame({column: values}, index=idx)


# ---------------------------------------------------------------- specs

def test_spec_name_prefixes_other_symbols():
    assert FeatureSpec("realized_vol").name == "realized_vol"
    assert FeatureSpec("realized_vol", symbol="ETHUSDT").name == "ETHUSDT:realized_vol"


def test_unknown_column_is_rejected_with_the_available_list():
    with pytest.raises(ValueError, match="unknown feature column 'rsi'.*realized_vol"):
        FeatureSpec("rsi")


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError, match="unknown normalization"):
        FeatureSpec("e_p", norm="minmax")


def test_parse_reads_config_entries():
    specs = parse_feature_specs([
        {"column": "e_p"},
        {"column": "log_return", "symbol": "ETHUSDT", "norm": "none"},
    ])
    assert specs == [FeatureSpec("e_p"), FeatureSpec("log_return", "ETHUSDT", "none")]


def test_parse_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate model features.*e_p"):
        parse_feature_specs([{"column": "e_p"}, {"column": "e_p", "norm": "none"}])


def test_other_symbols_lists_each_symbol_once_in_order():
    specs = [FeatureSpec("e_p"), FeatureSpec("realized_vol", "ETHUSDT"),
             FeatureSpec("log_return", "XRPUSD"), FeatureSpec("log_return", "ETHUSDT")]
    assert other_symbols(specs) == ["ETHUSDT", "XRPUSD"]


def test_every_documented_column_has_a_description():
    assert all(desc.strip() for desc in COLUMN_DESCRIPTIONS.values())


# ---------------------------------------------------------------- cross-asset alignment

def test_cross_asset_values_are_matched_on_the_same_bar():
    main = _table([0.1, 0.2, 0.3, 0.4])
    eth = _table([1.0, 2.0, 3.0, 4.0])
    out = attach_cross_asset_features(main, {"ETHUSDT": eth}, [FeatureSpec("realized_vol", "ETHUSDT")], missing="drop")
    assert out["ETHUSDT:realized_vol"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out["realized_vol"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_drop_policy_removes_bars_before_the_other_symbol_starts(capsys):
    main = _table([0.1, 0.2, 0.3, 0.4])
    eth = _table([3.0, 4.0], start="2024-01-01 02:00")
    out = attach_cross_asset_features(main, {"ETHUSDT": eth}, [FeatureSpec("realized_vol", "ETHUSDT")], missing="drop")
    assert list(out.index) == list(main.index[2:])
    assert "dropped 2 of 4 bars" in capsys.readouterr().out


def test_fill_policy_keeps_every_bar_and_flags_availability():
    main = _table([0.1, 0.2, 0.3, 0.4])
    eth = _table([3.0, 4.0], start="2024-01-01 02:00")
    out = attach_cross_asset_features(main, {"ETHUSDT": eth}, [FeatureSpec("realized_vol", "ETHUSDT")], missing="fill")
    assert list(out.index) == list(main.index)
    assert out["ETHUSDT:realized_vol"].isna().tolist() == [True, True, False, False]
    assert out["ETHUSDT:available"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_availability_requires_every_requested_column_of_the_symbol():
    main = _table([0.1, 0.2])
    eth = pd.DataFrame({"realized_vol": [1.0, 2.0], "log_return": [np.nan, 0.01]}, index=main.index)
    specs = [FeatureSpec("realized_vol", "ETHUSDT"), FeatureSpec("log_return", "ETHUSDT")]
    out = attach_cross_asset_features(main, {"ETHUSDT": eth}, specs, missing="fill")
    assert out["ETHUSDT:available"].tolist() == [0.0, 1.0]


def test_gaps_in_the_other_symbol_are_not_forward_filled():
    main = _table([0.1, 0.2, 0.3])
    eth = _table([1.0, 3.0]).set_axis(main.index[[0, 2]])
    out = attach_cross_asset_features(main, {"ETHUSDT": eth}, [FeatureSpec("realized_vol", "ETHUSDT")], missing="fill")
    assert np.isnan(out.loc[main.index[1], "ETHUSDT:realized_vol"])
    assert out.loc[main.index[1], "ETHUSDT:available"] == 0.0


def test_unknown_missing_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown missing policy"):
        attach_cross_asset_features(_table([0.1]), {}, [], missing="interpolate")


@pytest.mark.parametrize("missing", ["fill", "drop"])
def test_cross_asset_features_have_no_look_ahead(missing):
    """Truncating the other symbol's future must not change any earlier row."""
    main = _table(np.arange(10.0))
    eth = _table(np.arange(8.0) * 10, start="2024-01-01 02:00")
    spec = [FeatureSpec("realized_vol", "ETHUSDT")]
    full = attach_cross_asset_features(main, {"ETHUSDT": eth}, spec, missing=missing)
    cut = attach_cross_asset_features(main.iloc[:6], {"ETHUSDT": eth.iloc[:4]}, spec, missing=missing)
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_missing_symbol_table_raises():
    with pytest.raises(ValueError, match="no feature table loaded for symbol ETHUSDT"):
        attach_cross_asset_features(_table([0.1]), {}, [FeatureSpec("realized_vol", "ETHUSDT")])


def test_only_traded_symbol_features_leave_table_unchanged():
    main = _table([0.1, 0.2])
    pd.testing.assert_frame_equal(attach_cross_asset_features(main, {}, [FeatureSpec("realized_vol")]), main)


# ---------------------------------------------------------------- normalization

def _row(**values):
    return pd.Series(values)


def test_output_follows_spec_order():
    norm = FeatureNormalizer([FeatureSpec("e_i", norm="none"), FeatureSpec("e_p", norm="none")])
    assert norm.transform(_row(e_p=1.0, e_i=2.0)).tolist() == [2.0, 1.0]
    assert norm.names == ["e_i", "e_p"]


def test_zscore_is_zero_during_warmup_then_real():
    norm = FeatureNormalizer([FeatureSpec("e_p")], window=10, warmup=3)
    outs = [norm.transform(_row(e_p=v))[0] for v in [1.0, 2.0, 3.0]]
    assert outs[:2] == [0.0, 0.0]
    assert outs[2] == pytest.approx((3.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=1))


def test_zscore_uses_a_rolling_window():
    norm = FeatureNormalizer([FeatureSpec("e_p")], window=3, warmup=2)
    for v in [100.0, 1.0, 2.0]:
        norm.transform(_row(e_p=v))
    out = norm.transform(_row(e_p=3.0))[0]  # window now holds 1, 2, 3
    assert out == pytest.approx(1.0)


def test_none_passes_values_through():
    norm = FeatureNormalizer([FeatureSpec("log_return", norm="none")])
    assert norm.transform(_row(log_return=-0.03))[0] == -0.03


def test_reset_restarts_warmup():
    norm = FeatureNormalizer([FeatureSpec("e_p")], window=5, warmup=2)
    norm.transform(_row(e_p=1.0))
    norm.transform(_row(e_p=2.0))
    norm.reset()
    assert norm.transform(_row(e_p=5.0))[0] == 0.0


def test_missing_feature_raises():
    norm = FeatureNormalizer([FeatureSpec("realized_vol", "ETHUSDT")])
    with pytest.raises(ValueError, match="ETHUSDT:realized_vol"):
        norm.transform(_row(realized_vol=0.5))


def test_describe_lists_source_and_normalization():
    table = describe_features([FeatureSpec("e_p"), FeatureSpec("log_return", "ETHUSDT", "none")])
    assert table["input"].tolist() == ["e_p", "ETHUSDT:log_return"]
    assert table["source"].tolist() == ["traded symbol", "ETHUSDT"]
    assert table["norm"].tolist() == ["zscore", "none"]


def test_describe_names_the_traded_symbol_when_given():
    table = describe_features([FeatureSpec("e_p")], traded_symbol="BTCUSDT")
    assert table["source"].tolist() == ["BTCUSDT (traded)"]


def test_available_column_only_exists_for_other_symbols():
    with pytest.raises(ValueError, match="only exists for other symbols"):
        FeatureSpec("available")


def test_with_availability_flags_adds_one_raw_flag_per_other_symbol():
    specs = [FeatureSpec("e_p"), FeatureSpec("realized_vol", "ETHUSDT"), FeatureSpec("log_return", "ETHUSDT"),
             FeatureSpec("realized_vol", "XRPUSD")]
    out = with_availability_flags(specs)
    assert out[:4] == specs
    assert [s.name for s in out[4:]] == ["ETHUSDT:available", "XRPUSD:available"]
    assert all(s.norm == "none" for s in out[4:])


def test_with_availability_flags_keeps_an_explicitly_listed_flag():
    specs = [FeatureSpec("realized_vol", "ETHUSDT"), FeatureSpec("available", "ETHUSDT", norm="none")]
    assert with_availability_flags(specs) == specs


def test_missing_value_becomes_zero_and_does_not_touch_statistics():
    spec = [FeatureSpec("realized_vol", "ETHUSDT")]
    with_gap = FeatureNormalizer(spec, window=10, warmup=2)
    without_gap = FeatureNormalizer(spec, window=10, warmup=2)
    outs = [with_gap.transform(_row(**{"ETHUSDT:realized_vol": v}))[0] for v in [1.0, np.nan, 2.0, 3.0]]
    last = [without_gap.transform(_row(**{"ETHUSDT:realized_vol": v}))[0] for v in [1.0, 2.0, 3.0]][-1]
    assert outs[1] == 0.0
    assert outs[-1] == pytest.approx(last)
