"""Paired comparison of A2C-PID against static PID across walk-forward runs.

For every run directory, A2C-PID is averaged over its seeds per fold and
compared with static PID on the same fold. Reports, per metric, how many
folds A2C wins, an exact one-sided sign-test p-value, and the mean paired
difference with a bootstrap 95% confidence interval.

Run from repo root:
    python scripts/compare_runs.py results/run_*_A_*
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

# metric -> True if higher is better
METRICS = {
    "sharpe": True,
    "total_return": True,
    "max_drawdown": False,
    "vol_rmse": False,
    "invested_vol_error": False,
}


def sign_test_p(wins: int, n: int) -> float:
    """One-sided exact binomial p-value of at least `wins` successes out of n at p=0.5."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def invested_vol_error(run_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    """Adds |realized vol while invested - target_vol| per (fold, controller, seed).

    vol_rmse counts flat bars (direction 0) as zero vol, which under a trend
    direction mostly measures how often the controller was flat. This one
    only looks at bars with a nonzero position, so it measures sizing alone.
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    target = cfg["controller"]["target_vol"]
    bars_per_year = cfg["env"]["bars_per_year"]
    values = []
    for row in metrics.itertuples():
        name = row.controller_name if pd.isna(row.seed) else f"{row.controller_name}_s{int(row.seed)}"
        trace = pd.read_parquet(run_dir / "traces" / f"fold{int(row.fold_id):02d}_{name}.parquet")
        invested = trace.loc[trace["position"] != 0, "reward"].dropna()
        vol = invested.std(ddof=1) * np.sqrt(bars_per_year) if len(invested) > 24 else np.nan
        values.append(abs(vol - target))
    return metrics.assign(invested_vol_error=values)


CANDIDATES = ["a2c_pid", "a2c_online"]


def paired(metrics: pd.DataFrame, metric: str, higher_is_better: bool, candidate: str = "a2c_pid") -> dict:
    """Per-fold paired comparison of `candidate` (mean over seeds) against static_pid."""
    a2c = metrics[metrics.controller_name == candidate].groupby("fold_id")[metric].mean()
    static = metrics[metrics.controller_name == "static_pid"].set_index("fold_id")[metric]
    diff = (a2c - static.reindex(a2c.index)).dropna()
    if not higher_is_better:
        diff = -diff  # positive always means "A2C better"
    wins = int((diff > 0).sum())
    lo, hi = bootstrap_ci(diff.to_numpy())
    return {
        "folds": len(diff),
        "a2c_wins": wins,
        "sign_p": sign_test_p(wins, len(diff)),
        "mean_gain": float(diff.mean()),
        "ci95": f"[{lo:+.4f}, {hi:+.4f}]",
    }


def describe_run(run_dir: Path) -> str:
    """Short label of what a run varied: tag plus the settings that usually change."""
    cfg = json.loads((run_dir / "config.json").read_text())
    c = cfg["controller"]
    return (f"{cfg['data']['symbol']} fc={c['forecast_vol_window']} dir={c['direction']} "
            f"maxpos={cfg['env']['max_position']} w_return={cfg['training']['w_return']}")


def main(run_dirs: list[Path]) -> None:
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"skipping {run_dir}: no metrics.csv")
            continue
        metrics = invested_vol_error(run_dir, pd.read_csv(metrics_path))
        for candidate in CANDIDATES:
            if candidate not in set(metrics.controller_name):
                continue
            for metric, higher in METRICS.items():
                rows.append({"run": run_dir.name, "setup": describe_run(run_dir), "candidate": candidate,
                             "metric": metric, **paired(metrics, metric, higher, candidate)})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    for metric in METRICS:
        print(f"\n=== {metric} vs static_pid (mean_gain > 0 means candidate better; sign_p is one-sided)")
        print(table[table.metric == metric].drop(columns="metric").round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    main(parser.parse_args().run_dirs)
