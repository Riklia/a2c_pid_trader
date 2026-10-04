"""Generates diagnostic plots for a completed walk-forward run.

Reads metrics.csv and per-fold trace parquets from a run directory,
writes matplotlib figures back into the same directory.

Run from repo root:
    python scripts/plot_run.py results/run_20260713_143022
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


COLORS = {
    "buy_hold": "tab:gray",
    "vol_target": "tab:green",
    "static_pid": "tab:red",
    "a2c_pid": "tab:blue",
}


def load_fold_traces(run_dir: Path, fold_id: int) -> dict[str, pd.DataFrame]:
    """Returns {trace name: trace} for every controller (and seed) on one fold."""
    traces = {}
    for path in sorted((run_dir / "traces").glob(f"fold{fold_id:02d}_*.parquet")):
        name = path.stem.removeprefix(f"fold{fold_id:02d}_")
        traces[name] = pd.read_parquet(path)
    return traces


def _style(trace_name: str) -> dict:
    """a2c_pid seeds share one color, only the first gets a legend entry."""
    base = "a2c_pid" if trace_name.startswith("a2c_pid") else trace_name
    style = {"color": COLORS.get(base, "black")}
    if base == "a2c_pid":
        style["alpha"] = 0.6
        style["label"] = "a2c_pid (per seed)" if trace_name.endswith("_s0") else None
    else:
        style["label"] = trace_name
    return style


def plot_fold_comparison(run_dir: Path, fold_id: int, out_path: Path) -> None:
    """Draws equity, position, and cumulative log return for every controller on one fold."""
    traces = load_fold_traces(run_dir, fold_id)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Fold {fold_id}")

    for name, trace in traces.items():
        style = _style(name)
        axes[0].plot(trace.index, trace["equity"], **style)
        axes[1].plot(trace.index, trace["position"], **style)
        axes[2].plot(trace.index, trace["reward"].cumsum(), **style)

    axes[0].set_ylabel("equity")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("position")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("cumulative log return")
    axes[2].set_xlabel("time")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(run_dir: Path, out_path: Path) -> None:
    """Bar charts of return, Sharpe and vol_rmse per fold, a2c_pid averaged over seeds."""
    metrics = pd.read_csv(run_dir / "metrics.csv")

    panels = [("total_return", "total return"), ("sharpe", "Sharpe ratio"), ("vol_rmse", "rolling portfolio vol RMSE")]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 5))

    for ax, (column, title) in zip(axes, panels):
        pivot = metrics.pivot_table(index="fold_id", columns="controller_name", values=column, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, "black") for c in pivot.columns])
        ax.set_title(f"{title} per fold")
        ax.axhline(0.0, color="gray", linewidth=0.7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(run_dir: Path) -> None:
    if not run_dir.exists():
        raise SystemExit(f"run dir does not exist: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("plotting summary metrics...")
    plot_metrics_summary(run_dir, plots_dir / "summary.png")

    metrics = pd.read_csv(run_dir / "metrics.csv")
    fold_ids = sorted(metrics["fold_id"].unique())
    print(f"plotting {len(fold_ids)} fold comparisons...")
    for fold_id in fold_ids:
        plot_fold_comparison(run_dir, fold_id, plots_dir / f"fold{fold_id:02d}.png")

    print(f"\ndone, plots in {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    main(args.run_dir)
