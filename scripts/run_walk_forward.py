"""Walk-forward evaluation of A2CPID vs static PID on real Binance data.

Loads OHLCV and funding from Postgres, builds features, splits into train/test
folds, trains A2CPID online on each train window, evaluates both controllers
on each test window, and writes a per-fold metrics table to CSV.

Run from repo root:
    python scripts/run_walk_forward.py configs/walk_forward.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from backtest.walk_forward import (
    make_walk_forward_folds, evaluate_on_test, FoldMetrics,
)
from backtest.engine import BacktestResult
from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from data import loader, features
from environment.trading_env import TradingEnv
from models.actor import Actor
from models.critic import Critic
from models.pid_net import PIDNet
from training.train_loop import TrainConfig, train_one_pass
from data.cache import load_from_cache, save_to_cache, cache_path_for


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_dataset(cfg: dict) -> pd.DataFrame:
    # TODO: add cache version to the metadata, so no need to clean up manually after code changes
    cache_params = {
        "symbol": cfg["data"]["symbol"],
        "start_ts": cfg["data"]["start_ts"],
        "end_ts": cfg["data"]["end_ts"],
        "freq": cfg["data"]["freq"],
        "target_vol": cfg["features"]["target_vol"],
        "vol_window": cfg["features"]["vol_window"],
        "integral_mode": cfg["features"]["integral_mode"],
        "integral_window": cfg["features"]["integral_window"],
    }

    cached = load_from_cache(cache_params)
    if cached is not None:
        print(f"loaded {len(cached)} rows from cache")
        return cached

    dsn = os.environ.get("DEV_DB_DSN")
    if not dsn:
        raise SystemExit("DEV_DB_DSN not set")
    engine = loader.get_engine(loader.DBConfig(dsn=dsn))

    print(f"loading {cfg['data']['symbol']} from {cfg['data']['start_ts']} to {cfg['data']['end_ts']}...")
    trades = loader.load_trades(
        engine, symbol=cfg["data"]["symbol"],
        start_ts=cfg["data"]["start_ts"], end_ts=cfg["data"]["end_ts"],
    )
    funding = loader.load_funding(
        engine, symbol=cfg["data"]["symbol"],
        start_ts=cfg["data"]["start_ts"], end_ts=cfg["data"]["end_ts"],
    )
    print(f"loaded {len(trades)} trade rows, {len(funding)} funding rows")

    ft = features.build_feature_table(
        trades, funding,
        target_vol=cfg["features"]["target_vol"],
        freq=cfg["data"]["freq"],
        vol_window=cfg["features"]["vol_window"],
        integral_mode=cfg["features"]["integral_mode"],
        integral_window=cfg["features"]["integral_window"],
    )
    print(f"feature table: {len(ft)} rows from {ft.index[0]} to {ft.index[-1]}")

    save_to_cache(ft, cache_params)
    print(f"cached to {cache_path_for(cache_params)}")
    return ft


def make_static_pid(cfg: dict) -> StaticPID:
    return StaticPID(
        kp=cfg["static_pid"]["kp"],
        ki=cfg["static_pid"]["ki"],
        kd=cfg["static_pid"]["kd"],
    )


def make_a2c_pid(cfg: dict, seed: int) -> A2CPID:
    torch.manual_seed(seed)
    return A2CPID(
        static_pid=StaticPID(
            kp=cfg["a2c_pid"]["static_kp"],
            ki=cfg["a2c_pid"]["static_ki"],
            kd=cfg["a2c_pid"]["static_kd"],
        ),
        pid_net=PIDNet(
            hidden_sizes=tuple(cfg["a2c_pid"]["pid_net_hidden"]),
            gain_bounds=cfg["a2c_pid"]["gain_bounds"],
        ),
        actor=Actor(),
        critic=Critic(),
    )


def make_train_config(cfg: dict) -> TrainConfig:
    return TrainConfig(**cfg["training"])


def make_run_dir(base_dir: Path = REPO_ROOT / "results") -> Path:
    """Creates a fresh timestamped directory for this run's outputs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_result(result: BacktestResult, traces_dir: Path, fold_id: int, name: str):
    trace_df = pd.DataFrame({
        "equity": result.equity,
        "position": result.positions,
    })
    trace_df["reward"] = result.rewards  # different index, will align on join
    trace_df["action"] = result.actions
    trace_df.to_parquet(traces_dir / f"fold{fold_id:02d}_{name}.parquet")


def run(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    run_dir = make_run_dir()

    print(f"run dir: {run_dir}")

    data = build_dataset(cfg)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    folds = make_walk_forward_folds(
        data,
        train_size=cfg["walk_forward"]["train_size"],
        test_size=cfg["walk_forward"]["test_size"],
        step_size=cfg["walk_forward"]["step_size"],
    )
    print(f"\ncreated {len(folds)} walk-forward folds\n")

    fee_rate = cfg["env"]["fee_rate"]
    bars_per_year = cfg["env"]["bars_per_year"]
    target_vol = cfg["features"]["target_vol"]
    train_cfg = make_train_config(cfg)

    all_metrics: list[FoldMetrics] = []

    for fold in folds:
        print(f"fold {fold.fold_id}: train {fold.train_start.date()}..{fold.train_end.date()}, "
                f"test {fold.test_start.date()}..{fold.test_end.date()}")

        traces_dir = run_dir / "traces"
        traces_dir.mkdir(exist_ok=True)

        static_ctrl = make_static_pid(cfg)
        static_metrics, static_result = evaluate_on_test(
            fold, static_ctrl, "static_pid",
            target_vol=target_vol, fee_rate=fee_rate, bars_per_year=bars_per_year,
        )
        all_metrics.append(static_metrics)
        print(f"  static_pid: return={static_metrics.total_return:+.2%} "
                f"sharpe={static_metrics.sharpe:+.2f} "
                f"dd={static_metrics.max_drawdown:.2%} "
                f"rmse={static_metrics.vol_rmse:.4f}")
        save_result(static_result, traces_dir, fold.fold_id, "static_pid")

        a2c_ctrl = make_a2c_pid(cfg, seed=fold.fold_id)
        train_env = TradingEnv(fold.train_data, fee_rate=fee_rate)
        train_one_pass(train_env, a2c_ctrl, train_cfg, target_vol=target_vol)

        a2c_metrics, a2c_result = evaluate_on_test(
            fold, a2c_ctrl, "a2c_pid",
            target_vol=target_vol, fee_rate=fee_rate, bars_per_year=bars_per_year,
        )
        all_metrics.append(a2c_metrics)
        print(f"  a2c_pid:    return={a2c_metrics.total_return:+.2%} "
                f"sharpe={a2c_metrics.sharpe:+.2f} "
                f"dd={a2c_metrics.max_drawdown:.2%} "
                f"rmse={a2c_metrics.vol_rmse:.4f}")
        save_result(a2c_result, traces_dir, fold.fold_id, "a2c_pid")


    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])

    out_path = REPO_ROOT / cfg["output"]["results_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)
    print(f"\nwrote {len(metrics_df)} rows to {out_path}")

    print("\nsummary across folds:")
    print(
        metrics_df
        .groupby("controller_name")[["total_return", "sharpe", "max_drawdown", "vol_rmse"]]
        .agg(["mean", "median", "std"])
        .to_string()
    )

    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nwrote metrics to {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run(args.config)
