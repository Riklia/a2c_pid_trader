"""Walk-forward evaluation of A2CPID vs static PID on real Binance data.

Loads OHLCV and funding from Postgres, builds features, splits into train/test
folds, trains A2CPID online on each train window (once per seed), evaluates
all controllers on each test window, and writes a per-fold metrics table to CSV.

Controllers compared on every fold:
    buy_hold    constant long, the no-skill benchmark
    vol_target  plain vol targeting, i.e. the control law with kp=1, ki=kd=0
    static_pid  the control law with the configured static gains
    a2c_pid     static_pid's gains plus PIDNet's dynamic gains, one run per seed,
                trained on the train window and frozen during the test window
    a2c_online  the same trained controller, but it keeps learning online during
                the test window (a2c_pid.evaluate_online), as in the paper

Run from repo root:
    python scripts/run_walk_forward.py configs/walk_forward.json --workers 20
    python scripts/run_walk_forward.py configs/walk_forward.json --tag long_only \\
        --set controller.direction='"long_only"'
"""
import argparse
import copy
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from backtest.engine import BacktestResult
from backtest.walk_forward import (
    make_walk_forward_folds, evaluate_on_test, FoldMetrics, WalkForwardFold,
)
from control import tuning
from control.a2c_pid import A2CPID
from control.baselines import BuyAndHold
from control.controller_base import Controller
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw
from data import loader, features, model_features
from data.cache import load_from_cache, save_to_cache, cache_path_for
from environment.trading_env import TradingEnv
from models.pid_net import PIDNet, PIDNetInputBuilder
from training.online import run_online
from training.train_loop import TrainConfig, train_one_pass


FEATURES_VERSION = 2


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Applies "section.key=<json value>" overrides, e.g. controller.direction='"long_only"'."""
    for item in overrides:
        path, raw = item.split("=", 1)
        section, key = path.split(".", 1)
        if key not in cfg[section]:
            raise SystemExit(f"unknown config key: {path}")
        cfg[section][key] = json.loads(raw)
    return cfg


def config_for_symbol(cfg: dict, symbol: str) -> dict:
    """Same config, pointed at another symbol."""
    out = copy.deepcopy(cfg)
    out["data"]["symbol"] = symbol
    return out


# ---------------------------------------------------------------- data

def load_feature_table(cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """Per-symbol feature table, built from Postgres once and cached."""
    cache_params = {
        # bump whenever feature logic changes without a parameter change
        "features_version": FEATURES_VERSION,
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
        if verbose:
            print(f"loaded {cfg['data']['symbol']}: {len(cached)} rows from cache")
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


def symbol_table(cfg: dict, symbol: str, verbose: bool = True) -> pd.DataFrame:
    """Cached feature table of one symbol plus its sizing columns (vol_forecast, trend, mom_*).

    All direction signals are computed whichever one the config uses, so every
    direction variant drops the same warmup rows and gets identical folds.
    """
    c = cfg["controller"]
    return features.add_sizing_features(
        load_feature_table(config_for_symbol(cfg, symbol), verbose=verbose),
        freq=cfg["data"]["freq"],
        forecast_vol_window=c["forecast_vol_window"],
        trend_window=c["trend_window"],
        momentum_windows=tuple(c["momentum_windows"]),
    )


def model_feature_specs(cfg: dict) -> list[model_features.FeatureSpec]:
    """The configured model features, plus SYMBOL:available flags under the "fill" policy."""
    specs = model_features.parse_feature_specs(cfg["a2c_pid"]["features"])
    if cfg["a2c_pid"]["cross_asset_missing"] == "fill":
        specs = model_features.with_availability_flags(specs)
    return specs


def build_dataset(cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """Table of the traded symbol, plus the cross-asset model features
    (`SYMBOL:column`) the config asks for, aligned bar by bar."""
    specs = model_feature_specs(cfg)
    others = {symbol: symbol_table(cfg, symbol, verbose) for symbol in model_features.other_symbols(specs)}
    return model_features.attach_cross_asset_features(
        symbol_table(cfg, cfg["data"]["symbol"], verbose), others, specs,
        missing=cfg["a2c_pid"]["cross_asset_missing"], verbose=verbose,
    )


def make_folds(cfg: dict, data: pd.DataFrame) -> list[WalkForwardFold]:
    return make_walk_forward_folds(
        data,
        train_size=cfg["walk_forward"]["train_size"],
        test_size=cfg["walk_forward"]["test_size"],
        step_size=cfg["walk_forward"]["step_size"],
        window=cfg["walk_forward"].get("window", "rolling"),
    )


# ---------------------------------------------------------------- controllers

def resolve_static_gains(cfg: dict) -> dict[str, float]:
    """Static gains from the config, where "derived" means the value from control.tuning."""
    derived = tuning.derived_gains(cfg["controller"]["port_vol_window"])
    return {
        name: derived[name] if value == "derived" else float(value)
        for name, value in cfg["static_pid"].items()
    }


def make_law(cfg: dict) -> VolTargetLaw:
    """Builds a fresh control law; every controller needs its own, since the law is stateful."""
    c = cfg["controller"]
    ki = resolve_static_gains(cfg)["ki"]
    integral_limit = tuning.integral_limit_for(c["max_integral_correction"], ki) if ki > 0 else math.inf
    return VolTargetLaw(
        target_vol=c["target_vol"],
        max_position=cfg["env"]["max_position"],
        integral_limit=integral_limit,
        port_vol_window=c["port_vol_window"],
        bars_per_year=cfg["env"]["bars_per_year"],
        direction_mode=c["direction"],
        trend_column=c["trend_column"],
        direction_hold=c["direction_hold"],
    )


def make_static_pid(cfg: dict) -> StaticPID:
    return StaticPID(law=make_law(cfg), **resolve_static_gains(cfg))


def make_a2c_pid(cfg: dict, seed: int) -> A2CPID:
    """A2CPID on top of exactly the same static PID as the baseline, so any
    difference between the two comes from the learned dynamic gains."""
    torch.manual_seed(seed)
    normalizer = model_features.FeatureNormalizer(model_feature_specs(cfg))
    return A2CPID(
        static_pid=make_static_pid(cfg),
        pid_net=PIDNet(
            input_size=PIDNetInputBuilder.input_size(len(normalizer)),
            hidden_sizes=tuple(cfg["a2c_pid"]["pid_net_hidden"]),
            gain_bounds=tuple(cfg["a2c_pid"]["gain_bounds"]),
        ),
        normalizer=normalizer,
    )


def make_train_config(cfg: dict) -> TrainConfig:
    return TrainConfig(**cfg["training"], bars_per_year=cfg["env"]["bars_per_year"])


# ---------------------------------------------------------------- folds

def make_run_dir(base_dir: Path = REPO_ROOT / "results", tag: str | None = None) -> Path:
    """Creates a fresh timestamped directory for this run's outputs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / (f"run_{ts}" if tag is None else f"run_{ts}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_result(result: BacktestResult, traces_dir: Path, fold_id: int, name: str):
    trace_df = pd.DataFrame({
        "equity": result.equity,
        "position": result.positions,
    })
    trace_df["reward"] = result.rewards  # different index, will align on join
    trace_df["action"] = result.actions
    trace_df.to_parquet(traces_dir / f"fold{fold_id:02d}_{name}.parquet")


def format_metrics(m: FoldMetrics) -> str:
    return (f"return={m.total_return:+7.2%} sharpe={m.sharpe:+.2f} dd={m.max_drawdown:6.2%} "
            f"vol={m.ann_vol:.3f} rmse={m.vol_rmse:.3f} turnover={m.turnover:5.1f} "
            f"beta={m.beta:+.2f} alpha={m.alpha:+.2f}")


_WORKER: dict = {}


def _init_worker(cfg: dict) -> None:
    # one thread per process, parallelism comes from running folds side by side
    torch.set_num_threads(1)
    _WORKER["cfg"] = cfg
    _WORKER["folds"] = make_folds(cfg, build_dataset(cfg, verbose=False))


def run_fold(fold_id: int, traces_dir: Path) -> tuple[list[FoldMetrics], list[str]]:
    """Evaluates every controller on one fold. Returns metrics and log lines."""
    cfg = _WORKER["cfg"]
    fold = _WORKER["folds"][fold_id]
    eval_kwargs = dict(
        target_vol=cfg["controller"]["target_vol"],
        fee_rate=cfg["env"]["fee_rate"],
        bars_per_year=cfg["env"]["bars_per_year"],
        max_position=cfg["env"]["max_position"],
        vol_window=cfg["controller"]["port_vol_window"],
    )
    metrics: list[FoldMetrics] = []
    lines = [f"fold {fold.fold_id}: train {fold.train_start.date()}..{fold.train_end.date()}, "
             f"test {fold.test_start.date()}..{fold.test_end.date()}"]

    def evaluate(controller: Controller, name: str, seed: int | None = None, **kwargs) -> None:
        m, result = evaluate_on_test(fold, controller, name, seed=seed, **eval_kwargs, **kwargs)
        metrics.append(m)
        label = name if seed is None else f"{name}[s{seed}]"
        lines.append(f"  {label:14s} {format_metrics(m)}")
        save_result(result, traces_dir, fold.fold_id, name if seed is None else f"{name}_s{seed}")

    evaluate(BuyAndHold(), "buy_hold")
    evaluate(StaticPID(law=make_law(cfg), kp=1.0), "vol_target")
    evaluate(make_static_pid(cfg), "static_pid")

    for seed in cfg["a2c_pid"]["seeds"]:
        a2c_ctrl = make_a2c_pid(cfg, seed=fold.fold_id * 1000 + seed)
        train_env = TradingEnv(
            fold.train_data, fee_rate=cfg["env"]["fee_rate"], max_position=cfg["env"]["max_position"],
        )
        train_one_pass(train_env, a2c_ctrl, make_train_config(cfg))
        # copy first: the frozen backtest advances the controller's normalization state
        online_ctrl = copy.deepcopy(a2c_ctrl) if cfg["a2c_pid"]["evaluate_online"] else None
        evaluate(a2c_ctrl, "a2c_pid", seed=seed)
        if online_ctrl is not None:
            train_cfg = make_train_config(cfg)
            evaluate(online_ctrl, "a2c_online", seed=seed,
                     runner=lambda env, ctrl: run_online(env, ctrl, train_cfg))

    return metrics, lines


def run(cfg_path: Path, overrides: list[str] = (), tag: str | None = None, workers: int = 1) -> None:
    cfg = apply_overrides(load_config(cfg_path), list(overrides))
    run_dir = make_run_dir(tag=tag)

    print(f"run dir: {run_dir}")
    print(f"static gains: {resolve_static_gains(cfg)}, integral limit: {make_law(cfg).integral_limit:.1f} bars")
    direction = cfg["controller"]["direction"]
    print(f"direction: {direction}" + (f" on {cfg['controller']['trend_column']}" if direction != "long_only" else ""))
    print("\nA2C-PID model features (edit a2c_pid.features in the config, see docs/features.md):")
    print(model_features.describe_features(model_feature_specs(cfg), cfg["data"]["symbol"]).to_string(index=False))
    print()

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # builds (and caches) every table once before workers start reading them
    folds = make_folds(cfg, build_dataset(cfg))
    print(f"\ncreated {len(folds)} walk-forward folds, running on {workers} worker(s)\n")

    traces_dir = run_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    all_metrics: list[FoldMetrics] = []

    fold_ids = [fold.fold_id for fold in folds]
    if workers == 1:
        _init_worker(cfg)
        outputs = (run_fold(i, traces_dir) for i in fold_ids)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg,))
        outputs = executor.map(run_fold, fold_ids, [traces_dir] * len(fold_ids))
    try:
        for metrics, lines in outputs:  # in fold order
            all_metrics.extend(metrics)
            print("\n".join(lines), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])

    if tag is None:  # tagged runs are variants, keep the shared csv for the main config
        out_path = REPO_ROOT / cfg["output"]["results_csv"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path, index=False)
        print(f"\nwrote {len(metrics_df)} rows to {out_path}")

    print("\nsummary across folds (a2c_pid pools all seeds):")
    print(
        metrics_df
        .groupby("controller_name")[["total_return", "sharpe", "max_drawdown", "ann_vol", "beta", "alpha", "turnover"]]
        .agg(["mean", "median"])
        .round(3)
        .to_string()
    )

    per_fold = metrics_df.groupby(["fold_id", "controller_name"])["total_return"].mean().unstack()
    print("\ntotal return per fold (a2c_pid = mean over seeds):")
    print(per_fold.map(lambda x: f"{x:+.2%}").to_string())
    wins = int((per_fold["a2c_pid"] > per_fold["static_pid"]).sum())
    print(f"\na2c_pid beats static_pid on {wins}/{len(per_fold)} folds")

    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nwrote metrics to {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=JSON",
                        help="override a config value, repeatable")
    parser.add_argument("--tag", help="suffix for the run directory name")
    parser.add_argument("--workers", type=int, default=1, help="folds evaluated in parallel")
    args = parser.parse_args()
    run(args.config, args.overrides, args.tag, args.workers)
