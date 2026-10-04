# Debugging

How to answer the questions that come up when something looks wrong.

## "What does the controller actually see at step N?"

Open the env viewer:

```bash
streamlit run tools/env_viewer/app.py
```

Pick a symbol, date range, and features. Press Step to advance one bar.
The page shows the full state row the controller would receive, and a
running trace with everything a reward computation uses: action, position
before and after, equity before and after, next bar's log return, funding,
settlement flag.

## "Did the network actually learn, or did it just not fail?"

Pass an `on_step` callback to `train_one_pass` and watch the effective
gains (`report.kp`, `.ki`, `.kd`). If they never leave the static values,
the network is not learning (too low `lr`, too high `w_gain`, or no
signal in the loss).

Then open any trace parquet from a completed run and compare mean position
early vs late in the test window:

```python
import pandas as pd
t = pd.read_parquet("results/run_<ts>/traces/fold02_a2c_pid_s0.parquet")
first = t["position"].iloc[:len(t)//5].abs().mean()
last  = t["position"].iloc[-len(t)//5:].abs().mean()
print(first, "->", last)
```

If `last` is far from `first`, something changed during training. If
`last ≈ first ≈ 0`, the network found a passive local minimum.

Also check `range` and `std`. If `std < 0.05` across the whole test
window, the controller is basically inactive.

## "The result changed between runs, is this noise or a bug?"

A2C-PID is trained once per seed listed in `a2c_pid.seeds`, and each
seed is pinned, so a rerun of the same config reproduces the same numbers.
Compare the spread across seeds before reading anything into a
difference between A2C-PID and the static PID.

If the Sharpe swings from +2 to -2 between runs, the controller is
too sensitive to initialization and either the learning rate is too
high or the architecture is unstable.

## "I changed features, but the results did not change"

The feature table is cached under `.cache/features/` by a hash of the
parameters that affect it. If you change the logic *inside* a feature
computation without changing any parameter the cache key sees, you will
get stale data served back.

Fix: bump `FEATURES_VERSION` in `scripts/run_walk_forward.py`. It is part
of the cache key, so the next run rebuilds the table.

## "Is variant X better?"

Run it as a tagged variant instead of editing the config:

```bash
python scripts/run_walk_forward.py configs/walk_forward.json --tag trend     --set controller.direction='"trend_long_flat"'
```

Tagged runs write to `results/run_<timestamp>_<tag>/` and leave
`results/walk_forward_results.csv` alone, so several variants can run in
parallel.

Add `--workers N` to evaluate N folds in parallel processes (one torch
thread each). Results are identical to a sequential run, seeds are pinned
per fold.

## Where everything from a run lives

Every walk-forward invocation writes to `results/run_<timestamp>/`:

- `config.json`: exact config used, so the run is reproducible.
- `metrics.csv`: one row per (fold, controller, seed) with return,
  Sharpe, drawdown, realized vol, vol RMSE, turnover, beta, alpha.
- `traces/foldNN_<controller>.parquet`: raw per-bar equity, position,
  reward, action. A2C-PID traces carry the seed: `foldNN_a2c_pid_sK`.

Nothing is overwritten between runs. Diffing two runs is just diffing
two CSVs.
