import numpy as np
import pandas as pd

from data.validation import require_sorted_unique_datetime_index


HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

VOL_WINDOW = "24h"  # realized vol lookback
INTEGRAL_WINDOW = "720h"  # 30 days, rolling PID integral term


def resample_ohlcv(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "trades": "sum",
    }
    out = df.resample(freq).agg(agg)
    return out.dropna(subset=["close"])


def align_funding_to_ohlcv(ohlcv: pd.DataFrame, funding: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Forward-fills funding onto the ohlcv bar grid, without look-ahead.

    Args:
        ohlcv: resampled OHLCV data with a DatetimeIndex.
        funding: raw funding data with a DatetimeIndex, updated less frequently
            than `freq` (e.g. every 8h).
        freq: bar frequency to align onto, must match `ohlcv`'s index frequency.
    """
    funding_resampled = funding[["mark_price", "funding"]].resample(freq).last()
    is_settlement = funding_resampled["funding"].notna().reindex(ohlcv.index, fill_value=False)

    funding_filled = funding_resampled.ffill()
    aligned = funding_filled.reindex(ohlcv.index, method="ffill")
    aligned["is_funding_settlement"] = is_settlement

    return ohlcv.join(aligned)


def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    return np.log(df[price_col] / df[price_col].shift(1))


def _bars_per_year(freq: str) -> float:
    """Returns how many bars of size `freq` fit in a year, for annualizing vol."""
    bar_seconds = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds()
    year_seconds = DAYS_PER_YEAR * HOURS_PER_DAY * 3600
    return year_seconds / bar_seconds


def _window_bar_count(freq: str, window: str) -> int:
    """Returns how many bars of size `freq` fit into a time-offset window."""
    bar = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    win = pd.Timedelta(pd.tseries.frequencies.to_offset(window))
    if win < bar:
        raise ValueError(f"window {window} is shorter than one bar ({freq})")
    return int(win / bar)


def compute_realized_vol(returns: pd.Series, freq: str, window: str = VOL_WINDOW, annualize: bool = True) -> pd.Series:
    """Computes rolling std of log returns over a time-based window, optionally annualized.

    Args:
        returns: log returns series with a DatetimeIndex.
        freq: bar frequency of `returns` (e.g. "1h"), used for the annualization factor
            and to require a full window before producing a value.
        window: rolling window as a pandas offset string, e.g. "24h".
        annualize: scale by sqrt(bars per year) if True.
    """
    min_periods = _window_bar_count(freq, window)
    vol = returns.rolling(window, min_periods=min_periods).std()
    if annualize:
        vol = vol * np.sqrt(_bars_per_year(freq))
    return vol


def compute_pid_errors(
    realized_vol: pd.Series,
    target_vol: float,
    integral_mode: str = "rolling",
    integral_window: str = INTEGRAL_WINDOW,
) -> pd.DataFrame:
    """Builds the p/i/d error signals that feed the self-tuning PID network.

    Args:
        realized_vol: realized volatility series, one value per bar.
        target_vol: volatility level the controller should track.
        integral_mode: "rolling" sums the error over integral_window
            (bounded, resets as the window slides); "expanding" accumulates
            over the full history like a classic PID integral term.
        integral_window: rolling window as a pandas offset string, e.g. "720H".
            Only used when integral_mode is "rolling".
    """
    e_p = target_vol - realized_vol

    if integral_mode == "rolling":
        e_i = e_p.rolling(integral_window).mean()
    elif integral_mode == "expanding":
        e_i = e_p.expanding().mean()
    else:
        raise ValueError(f"unknown integral_mode: {integral_mode}")

    e_d = e_p.diff()
    return pd.DataFrame({"e_p": e_p, "e_i": e_i, "e_d": e_d})


def build_feature_table(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    target_vol: float,
    freq: str = "1h",
    vol_window: str = VOL_WINDOW,
    integral_mode: str = "rolling",
    integral_window: str = INTEGRAL_WINDOW,
) -> pd.DataFrame:
    require_sorted_unique_datetime_index(ohlcv, "build_feature_table.ohlcv")
    require_sorted_unique_datetime_index(funding, "build_feature_table.funding")

    resampled = resample_ohlcv(ohlcv, freq=freq)
    joined = align_funding_to_ohlcv(resampled, funding, freq=freq)
    returns = compute_log_returns(joined)
    realized_vol = compute_realized_vol(returns, freq=freq, window=vol_window)
    errors = compute_pid_errors(realized_vol, target_vol, integral_mode, integral_window)

    result = joined.assign(log_return=returns, realized_vol=realized_vol)
    result = result.join(errors)
    return result.dropna()