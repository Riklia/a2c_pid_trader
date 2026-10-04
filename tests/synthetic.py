"""Synthetic market data shared by controller, training and learning tests."""
import numpy as np
import pandas as pd


def make_market(
    n: int = 400,
    vol: float | np.ndarray = 0.5,
    drift: float = 0.0,
    seed: int = 0,
    vol_forecast: float | np.ndarray | None = None,
) -> pd.DataFrame:
    """Gaussian hourly log returns with a known annualized vol and drift.

    Args:
        n: number of bars.
        vol: true annualized vol, scalar or one value per bar.
        drift: annualized drift of log returns.
        seed: rng seed.
        vol_forecast: value of the vol_forecast column; defaults to the true
            vol, i.e. a perfect forecast.

    realized_vol is set to the true vol too, so tests do not depend on
    rolling-window warmup.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vol = np.broadcast_to(np.asarray(vol, dtype=float), n)
    log_returns = rng.normal(drift / 8760, vol / np.sqrt(8760))
    log_returns[0] = np.nan
    forecast = vol if vol_forecast is None else np.broadcast_to(np.asarray(vol_forecast, dtype=float), n)
    return pd.DataFrame({
        "close": 100 * np.exp(np.cumsum(np.nan_to_num(log_returns))),
        "log_return": log_returns,
        "funding": np.zeros(n),
        "is_funding_settlement": np.zeros(n, dtype=bool),
        "realized_vol": vol.copy(),
        "vol_forecast": forecast.copy(),
    }, index=idx)
