# Debugging & Common Tasks

## 1. Run a quick variant without editing JSON configs

Use `--tag` and `--set` to override any config key from the command line:

```bash
python scripts/run_walk_forward.py configs/walk_forward.json --workers 8 \
    --tag btc_2x_gh24 \
    --set env.max_position=2.0 \
    --set a2c_pid.gain_hold=24
```

This creates `results/run_<timestamp>_btc_2x_gh24/` and leaves `results/walk_forward_results.csv` untouched so you can run multiple experiments in parallel.

## 2. Compare runs statistically

To see how `a2c_pid` (frozen) and `a2c_online` performed against `static_pid` across all test quarters (win rate, sign-test p-value, and 95% bootstrap confidence interval):

```bash
python scripts/compare_runs.py results/run_<timestamp>_*
```

## 3. Check if `PIDNet` actually learned anything

Open a trace parquet from a finished run and compare `a2c_online` positions against `static_pid`:

```python
import pandas as pd

static = pd.read_parquet("results/run_<ts>/traces/fold05_static_pid.parquet")
online = pd.read_parquet("results/run_<ts>/traces/fold05_a2c_online_s0.parquet")

diff = (online["position"] - static["position"]).abs()
print("Mean |pos_online - pos_static|:", diff.mean())
print("Max  |pos_online - pos_static|:", diff.max())
```

- If `diff.max() ≈ 0`, the network stayed at zero adjustment (learning rate too low, `w_gain` too high, or bounds set to zero).
- You can also pass an `on_step` callback to `train_one_pass` in Python to inspect `report.kp`, `report.ki`, and `report.kd` bar by bar.

## 4. "I changed feature code, but results didn't change"

Computed feature tables are cached in `.cache/features/` by a hash of config parameters. If you edit the math inside `data/features.py` without changing config parameters, it will load the old cached parquet.

**Fix:** Delete `.cache/features/` or bump `FEATURES_VERSION` in [scripts/run_walk_forward.py](../scripts/run_walk_forward.py).

## 5. Inspect the environment step by step

Launch the interactive Streamlit viewer to step through bars one by one and inspect raw state, positions, fees, and funding:

```bash
streamlit run tools/env_viewer/app.py
```

## 6. What lives in `results/run_<timestamp>/`

- `config.json` — exact config (including `--set` overrides) used for the run.
- `metrics.csv` — one row per `(fold, controller, seed)` with `total_return`, `sharpe`, `max_drawdown`, `realized_vol`, `vol_rmse`, `invested_vol`, `invested_vol_error`, `turnover`, `beta`, `alpha`.
- `traces/foldNN_<controller>.parquet` — hourly bar-by-bar `equity`, `position`, `reward`, and `action`.
- `plots/` — generated when you run `python scripts/plot_run.py results/run_<timestamp>`.
