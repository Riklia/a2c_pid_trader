# a2c_pid_trader

A self-tuning PID controller that sizes crypto positions to keep portfolio volatility near a target (default: 35% annualized).

Inspired by Sharifi & Alasty ([2022][paper]), this project keeps an interpretable PID control law as the trading policy and uses a small neural network (`PIDNet`) to adjust the PID gains online as market volatility shifts.

[paper]: https://arxiv.org/abs/2307.01312

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env    # set DEV_DB_DSN for Postgres
```

Run a walk-forward backtest (1-year train, 3-month test sliding across history):

```bash
python scripts/run_walk_forward.py configs/walk_forward.json --workers 8
python scripts/plot_run.py results/run_<timestamp>
```

Other useful scripts:

```bash
python scripts/describe_features.py configs/walk_forward.json   # inspect model inputs
python scripts/compare_runs.py results/run_*                    # paired stats vs static PID
python scripts/synthetic_benchmark.py --seeds 5 --workers 8     # test on synthetic vol regimes
python scripts/grid_search.py configs/grid_search.json          # hyperparameter search
streamlit run tools/env_viewer/app.py                           # step through bars in a UI
```

## Documentation

- [docs/overview.md](docs/overview.md) — how the system works end to end
- [docs/control_formulation.md](docs/control_formulation.md) — the PID control law, static tuning, and online learning math
- [docs/results.md](docs/results.md) — benchmark results and how to reproduce them
- [docs/features.md](docs/features.md) — input features, normalization, and how to change them
- [docs/debugging.md](docs/debugging.md) — common issues and how to check if the model is learning
