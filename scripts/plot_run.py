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


def load_trace(run_dir: Path, fold_id: int, controller_name: str) -> pd.DataFrame:
    path = run_dir / "traces" / f"fold{fold_id:02d}_{controller_name}.parquet"
    return pd.read_parquet(path)


def plot_fold_comparison(run_dir: Path, fold_id: int, out_path: Path) -> None:
    """Draws equity, position, and reward for both controllers on one fold."""
    static = load_trace(run_dir, fold_id, "static_pid")
    a2c = load_trace(run_dir, fold_id, "a2c_pid")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Fold {fold_id}: static PID vs A2C-PID")

    axes[0].plot(static.index, static["equity"], label="static PID", color="tab:red")
    axes[0].plot(a2c.index, a2c["equity"], label="A2C-PID", color="tab:blue")
    axes[0].set_ylabel("equity")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(static.index, static["position"], label="static PID", color="tab:red", alpha=0.7)
    axes[1].plot(a2c.index, a2c["position"], label="A2C-PID", color="tab:blue", alpha=0.7)
    axes[1].set_ylabel("position")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.7)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(static.index, static["reward"].cumsum(), label="static PID cum reward", color="tab:red")
    axes[2].plot(a2c.index, a2c["reward"].cumsum(), label="A2C-PID cum reward", color="tab:blue")
    axes[2].set_ylabel("cumulative reward")
    axes[2].set_xlabel("time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(run_dir: Path, out_path: Path) -> None:
    """Bar chart of Sharpe and vol_rmse across folds, side by side per controller."""
    metrics = pd.read_csv(run_dir / "metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot_sharpe = metrics.pivot(index="fold_id", columns="controller_name", values="sharpe")
    pivot_sharpe.plot(kind="bar", ax=axes[0], color=["tab:blue", "tab:red"])
    axes[0].set_title("Sharpe ratio per fold")
    axes[0].axhline(0.0, color="gray", linewidth=0.7)
    axes[0].set_ylabel("Sharpe")
    axes[0].grid(True, alpha=0.3, axis="y")

    pivot_rmse = metrics.pivot(index="fold_id", columns="controller_name", values="vol_rmse")
    pivot_rmse.plot(kind="bar", ax=axes[1], color=["tab:blue", "tab:red"])
    axes[1].set_title("portfolio vol tracking RMSE per fold")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3, axis="y")

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
