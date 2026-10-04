"""Synthetic benchmark: when does self-tuning beat a fixed-gain PID?

Each scenario is a market with a known true vol, so the right exposure is
known exactly. Static PID, A2C-PID frozen after training, and A2C-PID learning
online are run on the same test series, and scored by how far the portfolio's
realized vol lands from the target.

Scenarios:
    correct_design   the static design is right (perfect forecast). Control
                     case: nothing to adapt to, A2C should tie.
    bias_appears     trained on an unbiased forecast, then the forecast starts
                     overstating vol 2x (e.g. a data or model change).
    bias_drift       the forecast bias grows gradually from 1x to 2x over the test.
    regime_switching true vol jumps between 0.3 / 0.6 / 1.2 and the forecast is a
                     realistic lagging 7-day estimate.

Run from repo root:
    python scripts/synthetic_benchmark.py --seeds 5 --workers 4
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest.engine import run_backtest
from backtest.metrics import rolling_portfolio_vol_rmse
from control import tuning
from control.a2c_pid import A2CPID
from control.static_pid import StaticPID
from control.vol_target import VolTargetLaw
from data.model_features import DEFAULT_FEATURES, FeatureNormalizer
from data.synthetic import lagged_vol_forecast, make_market, regime_switching_vol
from environment.trading_env import TradingEnv
from models.pid_net import PIDNet, PIDNetInputBuilder
from training.online import run_online
from training.train_loop import TrainConfig

TARGET_VOL = 0.25
PORT_VOL_WINDOW = 168
N_TRAIN = 4000
N_TEST = 4000
FEE = 4e-4


def _market(vol: np.ndarray, seed: int, forecast_scale: np.ndarray | float = 1.0,
            lagging_forecast: bool = False) -> pd.DataFrame:
    """Market whose features are all causal estimates, never the true vol.

    realized_vol is a 24-bar rolling estimate; vol_forecast is either the true
    vol times `forecast_scale` (a forecast with a known bias) or a lagging
    168-bar rolling estimate.
    """
    data = make_market(n=len(vol), vol=vol, seed=seed)
    lr = data["log_return"].to_numpy()
    data["realized_vol"] = lagged_vol_forecast(lr, 24)
    base = lagged_vol_forecast(lr, 168) if lagging_forecast else vol
    data["vol_forecast"] = base * forecast_scale
    return data


def scenario(name: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) markets for a scenario."""
    n = N_TRAIN + N_TEST
    if name == "correct_design":
        data = _market(np.full(n, 0.5), seed)
    elif name == "bias_appears":
        scale = np.r_[np.ones(N_TRAIN), np.full(N_TEST, 2.0)]
        data = _market(np.full(n, 0.5), seed, forecast_scale=scale)
    elif name == "bias_drift":
        scale = np.r_[np.ones(N_TRAIN), np.linspace(1.0, 2.0, N_TEST)]
        data = _market(np.full(n, 0.5), seed, forecast_scale=scale)
    elif name == "regime_switching":
        vol = regime_switching_vol(n, (0.3, 0.6, 1.2), mean_duration=400, seed=seed)
        data = _market(vol, seed, lagging_forecast=True)
    else:
        raise ValueError(f"unknown scenario {name}")
    data = data.iloc[1:]  # first bar has no return
    return data.iloc[: N_TRAIN - 1], data.iloc[N_TRAIN - 1:]


def make_law() -> VolTargetLaw:
    gains = tuning.derived_gains(PORT_VOL_WINDOW)
    return VolTargetLaw(target_vol=TARGET_VOL, port_vol_window=PORT_VOL_WINDOW,
                        integral_limit=tuning.integral_limit_for(1.5, gains["ki"]))


def make_static() -> StaticPID:
    return StaticPID(law=make_law(), **tuning.derived_gains(PORT_VOL_WINDOW))


def make_a2c(cfg: dict, seed: int) -> A2CPID:
    torch.manual_seed(seed)
    normalizer = FeatureNormalizer(DEFAULT_FEATURES)
    return A2CPID(
        static_pid=make_static(),
        pid_net=PIDNet(
            input_size=PIDNetInputBuilder.input_size(len(normalizer)),
            hidden_sizes=tuple(cfg["a2c_pid"]["pid_net_hidden"]),
            gain_bounds=tuple(cfg["a2c_pid"]["gain_bounds"]),
        ),
        normalizer=normalizer,
    )


def score(result, warmup: int = PORT_VOL_WINDOW) -> dict:
    rewards = result.rewards.iloc[warmup:]
    return {
        "vol_error": abs(rewards.std() * np.sqrt(8760) - TARGET_VOL),
        "vol_rmse": rolling_portfolio_vol_rmse(rewards, TARGET_VOL, PORT_VOL_WINDOW, 8760),
    }


def run_case(args: tuple[str, int]) -> list[dict]:
    name, seed = args
    torch.set_num_threads(1)
    cfg = json.loads((REPO_ROOT / "configs" / "walk_forward.json").read_text())
    train_cfg = TrainConfig(**cfg["training"], bars_per_year=8760)
    train, test = scenario(name, seed)

    a2c = make_a2c(cfg, seed)
    run_online(TradingEnv(train, fee_rate=FEE), a2c, train_cfg)  # training pass
    frozen, online = a2c, copy.deepcopy(a2c)

    results = {
        "static_pid": run_backtest(TradingEnv(test, fee_rate=FEE), make_static()),
        "a2c_frozen": run_backtest(TradingEnv(test, fee_rate=FEE), frozen),
        "a2c_online": run_online(TradingEnv(test, fee_rate=FEE), online, train_cfg),
    }
    return [{"scenario": name, "seed": seed, "controller": c, **score(r)} for c, r in results.items()]


SCENARIOS = ["correct_design", "bias_appears", "bias_drift", "regime_switching"]


def main(seeds: int, workers: int, out: Path | None) -> None:
    cases = [(name, seed) for name in SCENARIOS for seed in range(seeds)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [row for rows in ex.map(run_case, cases) for row in rows]
    df = pd.DataFrame(rows)
    if out is not None:
        df.to_csv(out, index=False)

    summary = df.pivot_table(index="scenario", columns="controller", values="vol_rmse", aggfunc="mean")
    wins = (df.pivot_table(index=["scenario", "seed"], columns="controller", values="vol_rmse")
            .pipe(lambda t: (t["a2c_online"] < t["static_pid"]).groupby("scenario").sum()))
    summary["online_beats_static"] = wins.astype(int).astype(str) + f"/{seeds}"
    print(f"rolling portfolio vol RMSE vs target {TARGET_VOL} on the test series (lower is better), "
          f"mean over {seeds} seeds:")
    print(summary.reindex(SCENARIOS).round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="optional CSV of per-seed scores")
    args = parser.parse_args()
    main(args.seeds, args.workers, args.out)
