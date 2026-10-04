"""Grid search over A2C-PID hyperparameters without touching the reported folds.

The walk-forward folds are split by time:
    tuning folds   the first `tuning_folds` test quarters. Every grid point is
                   trained and evaluated online on these, and scored by its mean
                   paired gain over static PID on `objective`.
    holdout folds  every later quarter. Only the winning grid point runs here,
                   once, with `holdout_seeds`. These are the numbers to report.

Hyperparameters chosen on 2021-2022 are thus applied to 2022-2026 unchanged,
the same way they would be chosen and then used live. Static PID has no
hyperparameters to tune (its gains are derived), so it is the fixed reference.

The holdout run is written as a normal run directory, so
scripts/compare_runs.py works on it.

Run from repo root:
    python scripts/grid_search.py configs/grid_search.json --workers 20
"""
import argparse
import copy
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest.walk_forward import evaluate_on_test
from environment.trading_env import TradingEnv
from scripts import run_walk_forward as wf
from training.online import run_online
from training.train_loop import train_one_pass


def grid_points(grid: dict[str, list]) -> list[dict[str, object]]:
    """Cartesian product of the grid, as {"section.key": value} dicts."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def apply_point(cfg: dict, point: dict[str, object]) -> dict:
    out = copy.deepcopy(cfg)
    for path, value in point.items():
        section, key = path.split(".", 1)
        if key not in out[section]:
            raise SystemExit(f"unknown config key in grid: {path}")
        out[section][key] = value
    return out


# ---------------------------------------------------------------- workers

_W: dict = {}


def _init(base_cfg: dict) -> None:
    torch.set_num_threads(1)
    _W["cfg"] = base_cfg
    _W["folds"] = wf.make_folds(base_cfg, wf.build_dataset(base_cfg, verbose=False))


def _eval_kwargs(cfg: dict) -> dict:
    return dict(
        target_vol=cfg["controller"]["target_vol"],
        fee_rate=cfg["env"]["fee_rate"],
        bars_per_year=cfg["env"]["bars_per_year"],
        max_position=cfg["env"]["max_position"],
        vol_window=cfg["controller"]["port_vol_window"],
    )


def static_task(fold_id: int) -> dict:
    """Static PID metrics on one fold; independent of the grid point."""
    cfg, fold = _W["cfg"], _W["folds"][fold_id]
    m, _ = evaluate_on_test(fold, wf.make_static_pid(cfg), "static_pid", **_eval_kwargs(cfg))
    return {"fold_id": fold_id, **{k: getattr(m, k) for k in ("sharpe", "total_return", "max_drawdown")}}


def a2c_task(args: tuple[int, dict, int, int]) -> dict:
    """Trains A2C-PID for one grid point on one fold and evaluates it online."""
    point_id, point, fold_id, seed = args
    cfg = apply_point(_W["cfg"], point)
    fold = _W["folds"][fold_id]
    ctrl = wf.make_a2c_pid(cfg, seed=fold_id * 1000 + seed)
    train_cfg = wf.make_train_config(cfg)
    train_env = TradingEnv(fold.train_data, fee_rate=cfg["env"]["fee_rate"], max_position=cfg["env"]["max_position"])
    train_one_pass(train_env, ctrl, train_cfg)
    m, _ = evaluate_on_test(fold, ctrl, "a2c_online", seed=seed, **_eval_kwargs(cfg),
                            runner=lambda env, c: run_online(env, c, train_cfg))
    return {"point_id": point_id, "fold_id": fold_id, "seed": seed,
            **{k: getattr(m, k) for k in ("sharpe", "total_return", "max_drawdown")}}


# ---------------------------------------------------------------- main

def tune(spec: dict, base_cfg: dict, workers: int) -> tuple[pd.DataFrame, list[dict], int]:
    points = grid_points(spec["grid"])
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(base_cfg,)) as ex:
        n_folds = len(wf.make_folds(base_cfg, wf.build_dataset(base_cfg, verbose=False)))
        tuning = list(range(spec["tuning_folds"]))
        if spec["tuning_folds"] >= n_folds:
            raise SystemExit(f"tuning_folds={spec['tuning_folds']} leaves no holdout out of {n_folds} folds")
        print(f"{len(points)} grid points x {len(tuning)} tuning folds x {len(spec['tuning_seeds'])} seed(s) "
              f"= {len(points) * len(tuning) * len(spec['tuning_seeds'])} training runs; "
              f"holdout: folds {len(tuning)}..{n_folds - 1}\n", flush=True)

        static = pd.DataFrame(list(ex.map(static_task, tuning))).set_index("fold_id")
        tasks = [(i, p, f, s) for i, p in enumerate(points) for f in tuning for s in spec["tuning_seeds"]]
        started = time.time()
        rows = []
        for done, row in enumerate(ex.map(a2c_task, tasks), start=1):
            rows.append(row)
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} runs, {(time.time() - started) / 60:.1f} min", flush=True)

    a2c = pd.DataFrame(rows).groupby(["point_id", "fold_id"])[["sharpe", "total_return", "max_drawdown"]].mean()
    objective = spec["objective"]
    sign = -1 if objective == "max_drawdown" else 1  # drawdown: lower is better
    gain = sign * (a2c[objective] - static[objective].reindex(a2c.index, level="fold_id"))
    scores = gain.groupby("point_id").agg(mean_gain="mean", folds_won=lambda g: int((g > 0).sum()))
    table = pd.concat([pd.DataFrame(points), scores], axis=1).sort_values("mean_gain", ascending=False)
    return table, points, n_folds


def holdout(spec: dict, best_cfg: dict, n_folds: int, workers: int, tag: str) -> Path:
    """Runs the full controller set with the chosen hyperparameters on the holdout folds only."""
    best_cfg = copy.deepcopy(best_cfg)
    best_cfg["a2c_pid"]["seeds"] = spec["holdout_seeds"]
    best_cfg["a2c_pid"]["evaluate_online"] = True
    run_dir = wf.make_run_dir(tag=tag)
    (run_dir / "config.json").write_text(json.dumps(best_cfg, indent=2))
    traces = run_dir / "traces"
    traces.mkdir()

    fold_ids = list(range(spec["tuning_folds"], n_folds))
    metrics = []
    with ProcessPoolExecutor(workers, initializer=wf._init_worker, initargs=(best_cfg,)) as ex:
        for fold_metrics, lines in ex.map(wf.run_fold, fold_ids, [traces] * len(fold_ids)):
            metrics.extend(fold_metrics)
            print("\n".join(lines), flush=True)
    pd.DataFrame([m.__dict__ for m in metrics]).to_csv(run_dir / "metrics.csv", index=False)
    return run_dir


def main(spec_path: Path, workers: int, tag: str) -> None:
    spec = json.loads(spec_path.read_text())
    base_cfg = wf.apply_overrides(wf.load_config(REPO_ROOT / spec["base_config"]), spec["base_overrides"])
    wf.build_dataset(base_cfg)  # warm the cache before workers start

    started = time.time()
    table, points, n_folds = tune(spec, base_cfg, workers)
    pd.set_option("display.width", 250)
    print(f"\ngrid ranked by mean paired {spec['objective']} gain of online A2C over static PID "
          f"on the {spec['tuning_folds']} tuning folds:")
    print(table.round(4).to_string(index=False))

    best = points[int(table.index[0])]
    print(f"\nchosen: {best}\ntuning took {(time.time() - started) / 60:.1f} min; running holdout...\n", flush=True)
    run_dir = holdout(spec, apply_point(base_cfg, best), n_folds, workers, tag)
    table.to_csv(run_dir / "grid_tuning.csv", index=False)
    print(f"\nholdout run: {run_dir}  (total {(time.time() - started) / 60:.1f} min)")
    print(f"paired statistics: python scripts/compare_runs.py {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--tag", default="grid_holdout")
    args = parser.parse_args()
    main(args.spec, args.workers, args.tag)
