import numpy as np
import pandas as pd

from data.validation import require_sorted_unique_datetime_index


HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

VOL_WINDOW = "24h"  # realized vol lookback
INTEGRAL_WINDOW = "720h"  # 30 days, rolling PID integral term
FORECAST_VOL_WINDOW = "168h"  # 7 days, vol estimate used for position sizing


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
        funding: raw funding snapshots with a DatetimeIndex, containing
            mark_price, funding, and time_left_seconds columns.
        freq: bar frequency to align onto, must match `ohlcv`'s index frequency.
    """
    resets = funding["time_left_seconds"].diff() > 0
    settlement_bars = funding.index[resets].floor(freq).unique()

    funding_resampled = funding[["mark_price", "funding"]].resample(freq).last().ffill()
    aligned = funding_resampled.reindex(ohlcv.index, method="ffill")
    aligned["is_funding_settlement"] = aligned.index.isin(settlement_bars)

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


def add_sizing_features(
    table: pd.DataFrame,
    freq: str = "1h",
    forecast_vol_window: str = FORECAST_VOL_WINDOW,
    trend_window: str | None = None,
    momentum_windows: tuple[str, ...] = (),
    fast_vol_window: str | None = None,
) -> pd.DataFrame:
    """Adds the columns the vol-targeting controllers size positions from.

    Kept separate from build_feature_table so they can be derived from a
    cached table without reloading raw data. All columns only use data up
    to each bar's close.

    Columns added:
        vol_forecast: annualized realized vol over forecast_vol_window. Slower
            and less noisy than realized_vol, so sizing on it keeps turnover low.
        log_vol_forecast: log(vol_forecast), captures volatility level and mean reversion.
        trend: close / rolling mean of close over trend_window, minus 1. Only
            added when trend_window is given.
        mom_<window>: log return over the last <window> (time-series
            momentum), one column per entry of momentum_windows.
        downside_vol, downside_share, parkinson_vol, vol_ratio, hl_range:
            added when fast_vol_window is given (high-SNR range & semi-vol).

    Rows where any added column is still warming up are dropped.
    """
    log_returns = compute_log_returns(table)
    vol_forecast = compute_realized_vol(log_returns, freq=freq, window=forecast_vol_window)
    out = table.assign(
        vol_forecast=vol_forecast,
        log_vol_forecast=np.log(vol_forecast.clip(lower=1e-8)),
    )
    if trend_window is not None:
        min_periods = _window_bar_count(freq, trend_window)
        sma = out["close"].rolling(trend_window, min_periods=min_periods).mean()
        out["trend"] = out["close"] / sma - 1.0
    added = ["vol_forecast"] + (["trend"] if trend_window is not None else [])
    for window in momentum_windows:
        lag = _window_bar_count(freq, window)
        out[f"mom_{window}"] = np.log(out["close"] / out["close"].shift(lag))
        added.append(f"mom_{window}")

    if fast_vol_window is not None:
        fast_n = _window_bar_count(freq, fast_vol_window)
        bpy = _bars_per_year(freq)
        downside_vol = np.sqrt((log_returns.clip(upper=0.0) ** 2).rolling(fast_vol_window, min_periods=fast_n).mean() * bpy)
        upside_vol = np.sqrt((log_returns.clip(lower=0.0) ** 2).rolling(fast_vol_window, min_periods=fast_n).mean() * bpy)
        out["downside_vol"] = downside_vol
        out["downside_share"] = downside_vol / (downside_vol + upside_vol + 1e-8)
        added.extend(["downside_vol", "downside_share"])

        if "high" in out.columns and "low" in out.columns:
            hl_range = np.log((out["high"] / out["low"]).clip(lower=1.0))
            park_var = (hl_range ** 2) / (4.0 * np.log(2.0))
            parkinson_vol = np.sqrt(park_var.rolling(fast_vol_window, min_periods=fast_n).mean() * bpy)
            out["hl_range"] = hl_range
            out["parkinson_vol"] = parkinson_vol
            out["vol_ratio"] = np.log(parkinson_vol.clip(lower=1e-8) / out["vol_forecast"].clip(lower=1e-8))
            added.extend(["hl_range", "parkinson_vol", "vol_ratio"])
        else:
            fast_cc = compute_realized_vol(log_returns, freq=freq, window=fast_vol_window)
            out["vol_ratio"] = np.log(fast_cc.clip(lower=1e-8) / out["vol_forecast"].clip(lower=1e-8))
            added.append("vol_ratio")

    return out.dropna(subset=added)


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
    # Price history often starts years before funding history. Bars before the
    # first funding snapshot are kept with zero funding rather than dropped,
    # otherwise the funding coverage silently truncates the whole dataset.
    result["funding"] = result["funding"].fillna(0.0)
    return result.dropna(subset=[c for c in result.columns if c != "mark_price"])