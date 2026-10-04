"""Model features: what PIDNet sees, where each value comes from, and how it is normalized.

This module is the single place that answers "which features does the model use".
The list itself lives in the config (`a2c_pid.features`), one entry per feature:

    {"column": "realized_vol", "norm": "zscore"}                       # traded symbol
    {"column": "realized_vol", "symbol": "ETHUSDT", "norm": "zscore"}  # another symbol

- column: a column of the per-symbol feature table (see COLUMN_DESCRIPTIONS for
  what is available and how each one is computed).
- symbol: omit for the traded symbol; set it to read the same column from
  another symbol's feature table (cross-asset information).
- norm: how the raw value is transformed before it reaches the network, one of
  NORMALIZATIONS.

Bars where another symbol has no data are handled per a2c_pid.cross_asset_missing
(see MISSING_POLICIES).

The control law does not use these features; it sizes from vol_forecast and its
own error signals, and the network always sees those errors too. See docs/features.md.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.scaling import RollingZScoreScaler
from data.validation import require_columns


AVAILABLE = "available"

# Every column a model feature may read. Adding a column here (and computing it
# in data.features.build_feature_table) is what makes it usable as a feature.
COLUMN_DESCRIPTIONS = {
    "log_return": "log(close / previous close), the bar's own return",
    "realized_vol": "annualized std of hourly log returns over features.vol_window",
    "e_p": "features.target_vol - realized_vol",
    "e_i": "mean of e_p over features.integral_window (rolling) or all history (expanding)",
    "e_d": "one-bar change of e_p",
    "funding": "last funding rate known at the bar's close, 0 before funding history starts",
    "vol_forecast": "annualized std of hourly log returns over controller.forecast_vol_window, "
                    "the volatility the control law sizes positions with",
    "trend": "close / mean close over controller.trend_window - 1",
    # not part of any feature table: added by attach_cross_asset_features
    AVAILABLE: "1 if this symbol has a value for the bar, else 0 (cross-asset only, added automatically)",
}

# How to treat bars where a cross-asset feature has no value.
MISSING_POLICIES = {
    "fill": "keep the bar; the value reaches the network as 0 and SYMBOL:available is 0",
    "drop": "drop the bar from the dataset for every controller",
}

NORMALIZATIONS = {
    "none": "raw value, unchanged",
    "zscore": "(x - mean) / std over the last 500 bars seen by the controller, "
              "0 for the first 50 bars; online, so it never looks ahead",
}


MOMENTUM_PREFIX = "mom_"


def column_description(column: str) -> str | None:
    """How a feature-table column is computed, None if it is not a known column.

    Momentum columns are named after their window (mom_720h, mom_2160h, ...),
    one per entry of controller.momentum_windows.
    """
    if column in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[column]
    if column.startswith(MOMENTUM_PREFIX):
        window = column[len(MOMENTUM_PREFIX):]
        return f"log return over the last {window} (time-series momentum); {window} must be in controller.momentum_windows"
    return None


@dataclass(frozen=True)
class FeatureSpec:
    """One model input: a feature-table column, its source symbol and its normalization."""
    column: str
    symbol: str | None = None
    norm: str = "zscore"

    def __post_init__(self):
        if column_description(self.column) is None:
            raise ValueError(
                f"unknown feature column {self.column!r}, available: {sorted(COLUMN_DESCRIPTIONS)} "
                f"or {MOMENTUM_PREFIX}<window>"
            )
        if self.norm not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization {self.norm!r}, available: {sorted(NORMALIZATIONS)}")
        if self.column == AVAILABLE and self.symbol is None:
            raise ValueError("'available' only exists for other symbols, the traded symbol always has a value")

    @property
    def name(self) -> str:
        """Column name in the assembled dataset: `column`, or `SYMBOL:column` for other symbols."""
        return self.column if self.symbol is None else f"{self.symbol}:{self.column}"


# Fallback when no list is configured: the market's short- and long-window vol.
DEFAULT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("realized_vol"),
    FeatureSpec("vol_forecast"),
)


def parse_feature_specs(items: list[dict]) -> list[FeatureSpec]:
    """Builds specs from the config list, rejecting duplicates."""
    specs = [FeatureSpec(**item) for item in items]
    names = [spec.name for spec in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate model features: {duplicates}")
    return specs


def other_symbols(specs: list[FeatureSpec]) -> list[str]:
    """Symbols, other than the traded one, whose feature tables the specs read from."""
    return list(dict.fromkeys(spec.symbol for spec in specs if spec.symbol is not None))


def with_availability_flags(specs: list[FeatureSpec]) -> list[FeatureSpec]:
    """Appends a SYMBOL:available flag for every other symbol that does not list one already.

    Under the "fill" policy a missing value reaches the network as 0, which is
    also a perfectly ordinary z-score; the flag is what lets the network tell
    "no data" from "average".
    """
    present = {spec.name for spec in specs}
    flags = [
        FeatureSpec(AVAILABLE, symbol, norm="none")
        for symbol in other_symbols(specs)
        if f"{symbol}:{AVAILABLE}" not in present
    ]
    return list(specs) + flags


def attach_cross_asset_features(
    main: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    specs: list[FeatureSpec],
    missing: str = "fill",
    verbose: bool = True,
) -> pd.DataFrame:
    """Adds `SYMBOL:column` columns from other symbols' feature tables to the main table.

    Values are matched on the exact bar timestamp. Both tables use the same bar
    convention (a bar is labeled by its start and every column is known at its
    close), so a matched value is exactly as old as the main bar's own values:
    no look-ahead, and no forward-filling that could hide gaps.

    Bars where a requested value is missing (the other symbol's history starts
    later, or it has a gap) are handled per `missing`, see MISSING_POLICIES:
    "fill" keeps them as NaN and adds a SYMBOL:available column (1/0) per
    symbol; "drop" removes them. Coverage is printed unless verbose is False.
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(f"unknown missing policy {missing!r}, available: {sorted(MISSING_POLICIES)}")
    out = main.copy()
    added: dict[str, list[str]] = {}
    for spec in specs:
        if spec.symbol is None or spec.column == AVAILABLE:
            continue
        if spec.symbol not in others:
            raise ValueError(f"no feature table loaded for symbol {spec.symbol}")
        out[spec.name] = others[spec.symbol][spec.column].reindex(out.index)
        added.setdefault(spec.symbol, []).append(spec.name)

    if not added:
        return out

    for symbol, names in added.items():
        available = out[names].notna().all(axis=1)
        first = out.index[available][0] if available.any() else None
        if verbose:
            print(f"cross-asset {symbol}: values for {available.mean():.0%} of bars, first at {first}")
        if missing == "fill":
            out[f"{symbol}:{AVAILABLE}"] = available.astype(float)

    if missing == "drop":
        all_names = [n for names in added.values() for n in names]
        keep = out[all_names].notna().all(axis=1)
        if verbose and not keep.all():
            print(f"cross-asset features: dropped {int((~keep).sum())} of {len(out)} bars; data now spans "
                  f"{out.index[keep][0]} .. {out.index[keep][-1]}")
        out = out[keep]
    return out


class FeatureNormalizer:
    """Turns a state row into the normalized feature vector, in spec order.

    Stateful and online: z-score statistics are updated with every row it sees,
    from that row and earlier ones only (see data.scaling.RollingZScoreScaler).
    A missing value (NaN, from a cross-asset symbol without data for the bar)
    comes out as 0 and leaves that feature's statistics untouched, so z-scores
    are only ever computed from real values.

    Args:
        specs: the model features.
        window: bars of history behind each z-score.
        warmup: bars before z-scores are real; earlier calls return 0.
    """

    def __init__(self, specs: list[FeatureSpec] | tuple[FeatureSpec, ...], window: int = 500, warmup: int = 50):
        self.specs = list(specs)
        # one scaler per feature, so a missing value can skip just its own update
        self._scalers = {
            spec.name: RollingZScoreScaler([spec.name], window_size=window, warmup_size=warmup)
            for spec in self.specs if spec.norm == "zscore"
        }

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def __len__(self) -> int:
        return len(self.specs)

    def transform(self, state: pd.Series) -> np.ndarray:
        require_columns(state, set(self.names), "FeatureNormalizer.transform")
        out = np.zeros(len(self.specs), dtype=np.float64)
        for i, spec in enumerate(self.specs):
            value = float(state[spec.name])
            if not math.isfinite(value):
                continue
            if spec.norm == "none":
                out[i] = value
            else:
                scaled = self._scalers[spec.name].transform(pd.Series({spec.name: value}))
                out[i] = float(scaled[spec.name])
        return out

    def reset(self) -> None:
        for scaler in self._scalers.values():
            scaler.reset()


def describe_features(
    specs: list[FeatureSpec] | tuple[FeatureSpec, ...],
    traded_symbol: str | None = None,
) -> pd.DataFrame:
    """Human-readable table of the model features, for logs and the describe script."""
    traded = f"{traded_symbol} (traded)" if traded_symbol else "traded symbol"
    return pd.DataFrame([
        {
            "input": spec.name,
            "source": spec.symbol or traded,
            "computed as": column_description(spec.column),
            "norm": spec.norm,
        }
        for spec in specs
    ])
