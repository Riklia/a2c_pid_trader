# Features

To see exactly what a config feeds the pipeline:

```bash
python scripts/describe_features.py configs/walk_forward.json
```

The walk-forward script prints the same table at the start of every run.

## Who reads what

| Consumer | Inputs | Normalization | Configurable |
|---|---|---|---|
| Control law (`static_pid`, and the static part of `a2c_pid`) | `vol_forecast` for size, `controller.trend_column` for direction, plus its own realized returns for the integral term | raw | via the `controller` section |
| PIDNet (dynamic gains) | `u(k-1)`, `u(k-2)` (own last positions), the law's errors `e`, `e_int`, `e_d`, then every entry of `a2c_pid.features` | positions and errors raw (already dimensionless), features per entry | features: yes |

## Where values come from

Every symbol gets the same feature table, built from its minute candles and
funding snapshots by `data.features.build_feature_table` and cached under
`.cache/features/`; the sizing columns (`vol_forecast`, `trend`, `mom_*`)
are added on top by `data.features.add_sizing_features`. Its columns usable as model features are listed, with how
each is computed, in `COLUMN_DESCRIPTIONS` in
[data/model_features.py](../data/model_features.py):

| Column | Computed as |
|---|---|
| `log_return` | log(close / previous close), the bar's own return |
| `realized_vol` | annualized std of hourly log returns over `features.vol_window` |
| `e_p` | `features.target_vol` − `realized_vol` |
| `e_i` | mean of `e_p` over `features.integral_window` (rolling) or all history (expanding) |
| `e_d` | one-bar change of `e_p` |
| `funding` | last funding rate known at the bar's close, 0 before funding history starts |
| `vol_forecast` | annualized std of hourly log returns over `controller.forecast_vol_window`, what the law sizes with |
| `trend` | close / mean close over `controller.trend_window` − 1 |
| `mom_<window>` | log return over the last `<window>`, one column per `controller.momentum_windows` entry |
| `available` | other symbols only, added automatically under the `fill` policy: 1 if the symbol has a value for the bar, else 0 |

Every value in a bar's row is known at that bar's close. The environment
applies the chosen position to the *next* bar's return, so no feature looks
ahead.

## The feature list

`a2c_pid.features` in the config is the full, ordered list of PIDNet features:

```json
"features": [
  {"column": "realized_vol", "norm": "zscore"},
  {"column": "e_p", "norm": "zscore"},
  {"column": "realized_vol", "symbol": "ETHUSDT", "norm": "zscore"}
]
```

- `column`: a column from the table above.
- `symbol` (optional): omit it for the traded symbol (`data.symbol`, e.g.
  BTCUSDT); set it to read the column from another symbol's feature table.
  The feature is then named `SYMBOL:column`.
- `norm`: `zscore` (default) or `none`.

The network's input size follows the list automatically
(2 positions + 3 law errors + number of features + availability flags,
see below).

## Normalization

| `norm` | Transform |
|---|---|
| `zscore` | `(x − mean) / std` over the last 500 bars the controller has seen; 0 for the first 50 bars. Online, so it never uses future bars. Statistics carry over from training into the test window. |
| `none` | raw value |

Use `zscore` for anything whose scale differs across regimes or symbols (vols,
errors, returns); `none` only for values that are already on a unit scale.

## Cross-asset features

Other symbols' values are matched to the traded symbol's bars on the exact bar
timestamp; both tables label a bar by its start and know all its values at its
close, so a matched value is never newer than the traded bar. Nothing is
forward-filled.

Other symbols often start later than the traded one (ETH 2019-11, XRP/ADA/BNB
2020-07..09, against BTC 2019-09) or have gaps. `a2c_pid.cross_asset_missing`
decides what happens on those bars:

| policy | effect |
|---|---|
| `fill` (default) | the bar is kept. The missing value reaches the network as 0 and does not enter its z-score statistics; a `SYMBOL:available` input (1/0) is added automatically for every other symbol, so the network can tell "no data" from "average". |
| `drop` | the bar is removed for every controller, so the dataset starts where the latest symbol starts. |

`fill` keeps the traded symbol's whole history at the cost of extra inputs;
`drop` keeps inputs clean at the cost of history. Either way the run prints
each symbol's coverage.

## How to remove a feature

Delete its entry from `a2c_pid.features`.

## How to add a feature

- **An existing column, for the traded or another symbol:** add an entry to
  `a2c_pid.features`. Nothing else. A new momentum window needs its window
  in `controller.momentum_windows` too.
- **A new column:**
  1. Compute it in `data.features.build_feature_table`, using only data up to
     each bar's close (`tests/unit/test_features.py` has a look-ahead test to
     copy).
  2. Describe it in `COLUMN_DESCRIPTIONS` in `data/model_features.py`. A
     feature whose column is not described there is rejected at startup.
  3. Bump `FEATURES_VERSION` in `scripts/run_walk_forward.py` so cached
     tables are rebuilt.
  4. Add it to `a2c_pid.features`.
- **A new normalization:** add it to `NORMALIZATIONS` and implement it in
  `FeatureNormalizer`.
