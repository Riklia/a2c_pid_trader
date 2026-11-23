# Features

To see the exact inputs a config feeds into the controller and neural network:

```bash
python scripts/describe_features.py configs/walk_forward.json
```

## Who uses which inputs

1. **PID Control Law (`StaticPID` & base of `A2CPID`):**
   - `vol_forecast` (7-day realized vol by default) for position sizing.
   - `controller.trend_column` (`mom_720h`, 30-day log return) for direction (`+1` or `0`).
   - Its own realized portfolio returns over the last 168 invested bars for the integral term.
2. **Gain Network (`PIDNet`):**
   - Always receives 5 raw, dimensionless control signals: its last two positions (`u(k-1)`, `u(k-2)`) and the three PID errors (`e`, `e_int / e_limit`, `e_d`).
   - Followed by every feature listed in `a2c_pid.features` in the config.

## Available feature columns

Feature tables are built from 1-minute candles and funding snapshots (`data/features.py`) and cached in `.cache/features/`:

| Column | Meaning |
|---|---|
| `log_return` | 1-hour log return, `log(close / prev_close)` |
| `realized_vol` | Annualized std of hourly log returns over `features.vol_window` (default `24h`) |
| `vol_forecast` | Annualized std of hourly log returns over `controller.forecast_vol_window` (default `168h`) |
| `log_vol_forecast` | `log(vol_forecast)` |
| `parkinson_vol` | High-low range Parkinson volatility over `features.vol_window` |
| `vol_ratio` | `log(parkinson_vol / vol_forecast)` (fast-to-slow vol ratio) |
| `downside_vol` | Negative-return semi-volatility over `features.vol_window` |
| `downside_share` | `downside_vol / (downside_vol + upside_vol)` |
| `hl_range` | `log(high / low)` of the current bar |
| `funding` | Last known perpetual funding rate at bar close (`0` before funding history starts) |
| `trend` | `close / SMA(close, trend_window) - 1` |
| `mom_<window>` | Cumulative log return over `<window>` (e.g. `mom_720h`, `mom_2160h`), defined in `controller.momentum_windows` |
| `e_p`, `e_i`, `e_d` | Legacy linear error columns (`target_vol - realized_vol`) |

## Configuring `a2c_pid.features`

In your JSON config (or via `--set`), list the columns you want `PIDNet` to see:

```json
"features": [
  {"column": "realized_vol", "norm": "zscore"},
  {"column": "vol_forecast", "norm": "zscore"},
  {"column": "log_return", "norm": "zscore"},
  {"column": "funding", "norm": "zscore"},
  {"column": "mom_720h", "norm": "zscore"},
  {"column": "mom_2160h", "norm": "zscore"}
]
```

- **`norm`:**
  - `"zscore"` (default): Online rolling z-score `(x - mean) / std` over the last 500 bars (returns `0` for the first 50 warmup bars). Never looks ahead; stats carry over from training into the test quarter.
  - `"none"`: Raw value.
- **`symbol` (optional, for cross-asset inputs):** Add `"symbol": "ETHUSDT"` to pull a feature from another coin on matching timestamps. When `a2c_pid.cross_asset_missing` is `"fill"` (default), missing early bars are filled with `0` and an extra `SYMBOL:available` (`1` or `0`) flag is automatically appended. (Note: in our benchmarks, single-asset features performed slightly better out-of-sample than cross-asset features.)

## How to add a new feature column

1. Compute the column in `data/features.py` (inside `build_feature_table` or `add_sizing_features`) using only data up to each bar's close.
2. Add a one-line description to `COLUMN_DESCRIPTIONS` in [data/model_features.py](../data/model_features.py).
3. Increment `FEATURES_VERSION` in [scripts/run_walk_forward.py](../scripts/run_walk_forward.py) so cached feature tables rebuild automatically.
4. Add the column name to `a2c_pid.features` in your config.
