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

Open any trace parquet from a completed run and compare mean position
early vs late in the test window:

```python
import pandas as pd
t = pd.read_parquet("results/run_<ts>/traces/fold02_a2c_pid.parquet")
first = t["position"].iloc[:len(t)//5].abs().mean()
last  = t["position"].iloc[-len(t)//5:].abs().mean()
print(first, "->", last)
```

If `last` is far from `first`, something changed during training. If
`last ≈ first ≈ 0`, the network found a passive local minimum.

Also check `range` and `std`. If `std < 0.05` across the whole test
window, the controller is basically inactive.

## "The result changed between runs, is this noise or a bug?"

Seeds are not pinned by default. The paper's method is online and
non-deterministic by construction. Two consecutive runs on the same
config will differ, but the direction of change should be similar.

If the Sharpe swings from +2 to -2 between runs, the controller is
too sensitive to initialization and either the learning rate is too
high or the architecture is unstable.

## "I changed features, but the results did not change"

The feature table is cached under `.cache/features/` by a hash of the
parameters that affect it. If you change the logic *inside* a feature
computation without changing any parameter the cache key sees, you will
get stale data served back.

Fix: clear `.cache/features/` manually, or bump a `features_version`
constant that participates in the cache key.

This is a known rough edge. A cleaner fix would be to version the module
automatically, but we chose simplicity.

## Where everything from a run lives

Every walk-forward invocation writes to `results/run_<timestamp>/`:

- `config.json`: exact config used, so the run is reproducible.
- `metrics.csv`: one row per (fold, controller) with return, Sharpe,
  drawdown, vol RMSE.
- `traces/foldNN_<controller>.parquet`: raw per-bar equity, position,
  reward, action.
- `checkpoints/foldNN_a2c_pid.pt`: trained weights, scaler window, and
  input builder history. Load back with `A2CPID.load(path)`.

Nothing is overwritten between runs. Diffing two runs is just diffing
two CSVs.
