import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch

from control.a2c_pid import A2CPID
from environment.trading_env import TradingEnv
from training.losses import step_utility, trading_costs, vol_tracking_loss


@dataclass
class TrainConfig:
    """Hyperparameters governing one training run.

    The per-bar loss, with utility and costs annualized by bars_per_year:
        - w_return * utility + costs + w_track * vol_tracking + w_gain * gain_penalty

    Attributes:
        lr: Adam learning rate for pid_net.
        w_return: weight of the mean-variance utility. Return-driven gradients
            are very noisy, a year of hourly bars carries little signal.
        w_track: weight of the QLIKE vol-tracking loss.
        risk_aversion: variance penalty weight in the step utility.
        w_gain: L2 penalty on the dynamic gains relative to their bounds,
            keeps the controller close to its static PID unless the data
            argues otherwise.
        update_every: bars of utility gradient accumulated per optimizer step.
            Single-bar returns are almost pure noise, averaging over a batch
            is what makes the gradient informative.
        grad_clip: max global grad norm per optimizer step.
        epochs: passes over the training window. Each pass starts a fresh
            episode, weights carry over.
        bars_per_year: annualizes the utility so loss magnitudes are readable.
    """
    lr: float = 1e-3
    w_return: float = 1.0
    w_track: float = 1.0
    risk_aversion: float = 2.0
    w_gain: float = 1.0
    update_every: int = 24
    grad_clip: float = 1.0
    epochs: int = 1
    bars_per_year: float = 8760


@dataclass
class StepReport:
    step: int
    ts: pd.Timestamp
    action: float
    utility: float
    loss: float
    kp: float
    ki: float
    kd: float
    asset: str = ""


def train_one_pass(
    env: TradingEnv,
    controller: A2CPID,
    config: TrainConfig,
    on_step: Callable[[StepReport], None] | None = None,
) -> list[StepReport]:
    """Runs the controller through the env `config.epochs` times, updating pid_net online.

    Single-asset case of train_interleaved, see there for the details.
    """
    return train_interleaved({"": (env, controller)}, config, on_step=on_step)


def train_interleaved(
    streams: dict[str, tuple[TradingEnv, A2CPID]],
    config: TrainConfig,
    on_step: Callable[[StepReport], None] | None = None,
) -> list[StepReport]:
    """Trains one shared pid_net on several assets at once, walking them in calendar order.

    Each asset has its own env and controller (own law, scaler and position
    history, since those are per-asset state), but all controllers must
    share the same pid_net object. At every timestamp, each asset that has a
    bar there takes one step: the executed action is rebuilt with gradient
    (identical value, see A2CPID.differentiable_action) and scored against
    the next bar's return (see TrainConfig for the loss). Losses from all
    assets are accumulated over `update_every` timestamps, then pid_net
    takes one Adam step. Interleaving in calendar order, rather than one
    asset after another, keeps the network from drifting toward whichever
    asset it saw last, and never lets it see one asset's future while
    acting on another.

    Since the market does not react to the position, this pathwise gradient
    is the exact gradient of the realized loss with respect to the weights.

    Args:
        streams: {asset name: (fresh env, controller)}.
        config: hyperparameters for the loss and optimizer.
        on_step: optional callback receiving each StepReport as it happens,
            useful for logging without materializing the whole list.
    """
    nets = {id(controller.pid_net) for _, controller in streams.values()}
    if len(nets) != 1:
        raise ValueError("all controllers in train_interleaved must share one pid_net")
    pid_net = next(iter(streams.values()))[1].pid_net

    optimizer = torch.optim.Adam(pid_net.parameters(), lr=config.lr)
    # a decision is taken on every bar except each env's last one
    timeline = sorted(set().union(*(env.data.index[:-1] for env, _ in streams.values())))
    reports: list[StepReport] = []

    for _ in range(config.epochs):
        for env, controller in streams.values():
            env.reset()
            controller.reset()
        pending: list[torch.Tensor] = []

        for i, ts in enumerate(timeline):
            for name, (env, controller) in streams.items():
                if env.done or env.data.index[env.current_step] != ts:
                    continue
                step_loss, report = _train_step(env, controller, config, name)
                pending.append(step_loss)
                reports.append(report)
                if on_step is not None:
                    on_step(report)

            if pending and ((i + 1) % config.update_every == 0 or i == len(timeline) - 1):
                loss = torch.stack(pending).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pid_net.parameters(), config.grad_clip)
                optimizer.step()
                pending = []

    return reports


def _train_step(
    env: TradingEnv,
    controller: A2CPID,
    config: TrainConfig,
    asset: str,
) -> tuple[torch.Tensor, StepReport]:
    """Acts on the current bar, steps the env, returns the differentiable loss and a report."""
    state = env.data.iloc[env.current_step]
    prev_position = env.position
    action = controller.act(state)  # no-grad inference, advances controller state

    env.step(action)
    next_state = env.data.iloc[env.current_step]

    position, dyn_gains = controller.differentiable_action()
    funding = (
        float(next_state["funding"])
        if env.funding_enabled and next_state["is_funding_settlement"] else 0.0
    )
    simple_return = math.expm1(float(next_state["log_return"]))
    utility = step_utility(position, simple_return, risk_aversion=config.risk_aversion)
    costs = trading_costs(position, prev_position, funding_rate=funding, fee_rate=env.fee_rate)
    gain_penalty = ((dyn_gains / controller.pid_net.gain_bounds) ** 2).sum()
    step_loss = (
        (-config.w_return * utility + costs) * config.bars_per_year
        + config.w_gain * gain_penalty
    )
    if config.w_track and action != 0.0:
        step_loss = step_loss + config.w_track * vol_tracking_loss(
            position, simple_return, controller.law.target_vol, config.bars_per_year,
        )

    kp_d, ki_d, kd_d = (float(g) for g in dyn_gains.detach())
    report = StepReport(
        step=env.current_step,
        ts=next_state.name,
        action=action,
        utility=float(utility.detach()),
        loss=float(step_loss.detach()),
        kp=controller.static_pid.kp + kp_d,
        ki=controller.static_pid.ki + ki_d,
        kd=controller.static_pid.kd + kd_d,
        asset=asset,
    )
    return step_loss, report
