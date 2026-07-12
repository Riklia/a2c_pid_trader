from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import torch

from control.a2c_pid import A2CPID
from environment.trading_env import TradingEnv
from training.losses import compute_reward, td_error, actor_loss, critic_loss


@dataclass
class TrainConfig:
    """Hyperparameters governing one training run.

    Attributes:
        gamma: discount factor for TD error.
        lr: shared Adam learning rate for all four networks.
        w_error, w_error_rate, w_action: reward function weights (Eq. 10).
        w_prediction, w_entropy: actor loss term weights (Eq. 11a).
        w_critic: critic loss weight (Eq. 11b).
        eta: floor added to |TD| inside actor loss to keep exploration alive.
    """
    gamma: float = 0.99
    lr: float = 1e-3
    w_error: float = 1.0
    w_error_rate: float = 0.0
    w_action: float = 0.1
    w_prediction: float = 1.0
    w_entropy: float = 0.01
    w_critic: float = 1.0
    eta: float = 1e-3


@dataclass
class StepReport:
    step: int
    ts: pd.Timestamp
    action: float
    reward: float
    td: float
    actor_loss: float
    critic_loss: float
    predicted_vol: float
    actual_vol: float


def _actor_critic_input(state: pd.Series, action: float) -> torch.Tensor:
    """Assembles the 3-dim [u(k), s(k-1), s(k-2)] input, currently mocked
    by using [action, realized_vol, realized_vol] as a placeholder.

    TODO: wire this to a proper 2-step history mirror of PIDNetInputBuilder
    once we validate the training loop end-to-end. For now, both s slots
    use the same current realized_vol, which is a known simplification.
    """
    s = float(state["realized_vol"])
    return torch.tensor([action, s, s], dtype=torch.float32)


def train_one_pass(
    env: TradingEnv,
    controller: A2CPID,
    config: TrainConfig,
    target_vol: float,
    on_step: Callable[[StepReport], None] | None = None,
) -> list[StepReport]:
    """Runs a single online pass through the env, updating all networks in place.

    Args:
        env: fresh environment, not yet stepped.
        controller: A2CPID whose static gains, pid_net, actor and critic
            will be updated online.
        config: hyperparameters for reward, losses and optimizer.
        target_vol: reference volatility the controller tracks; used both
            in the reward formula and to interpret e_p in state rows.
        on_step: optional callback receiving each StepReport as it happens,
            useful for logging without materializing the whole list.
    """
    env.reset()

    optimizer = torch.optim.Adam(
        list(controller.pid_net.parameters())
        + list(controller.actor.parameters())
        + list(controller.critic.parameters()),
        lr=config.lr,
    )

    reports: list[StepReport] = []
    prev_realized_vol = float(env.data.iloc[0]["realized_vol"])
    prev_action = 0.0

    while not env.done:
        state = env.data.iloc[env.current_step]
        action = controller.act(state)  # no-grad inference

        _, _env_reward, _ = env.step(action)
        next_state = env.data.iloc[env.current_step]
        next_realized_vol = float(next_state["realized_vol"])

        controller.record_observation(position=action, realized_vol=next_realized_vol)

        reward = compute_reward(
            realized_vol=next_realized_vol, target_vol=target_vol,
            realized_vol_prev=prev_realized_vol, target_vol_prev=target_vol,
            action=action, action_prev=prev_action,
            w_error=config.w_error, w_error_rate=config.w_error_rate,
            w_action=config.w_action,
        )

        pid_input, _ = controller.build_training_inputs()
        pid_input_grad = pid_input.detach().clone().requires_grad_(True)
        dyn_gains = controller.pid_net(pid_input_grad)

        # rebuild u with grad so ∂u/∂θ_pid_net flows via Eq. 16 of the paper
        kp = controller.static_pid.kp + dyn_gains[0]
        ki = controller.static_pid.ki + dyn_gains[1]
        kd = controller.static_pid.kd + dyn_gains[2]
        u_grad = kp * state["e_p"] + ki * state["e_i"] + kd * state["e_d"]

        ac_input = _actor_critic_input(state, action)
        ac_input_next = _actor_critic_input(next_state, action)

        mu, sigma = controller.actor(ac_input)
        v_curr = controller.critic(ac_input)
        v_next = controller.critic(ac_input_next)

        td = td_error(reward=reward, value_next=v_next, value_current=v_curr, gamma=config.gamma)
        l_actor = actor_loss(
            predicted_next_state=mu, actual_next_state=next_realized_vol,
            sigma=sigma, td=td,
            w_prediction=config.w_prediction, w_entropy=config.w_entropy, eta=config.eta,
        )
        l_critic = critic_loss(td, weight=config.w_critic)

        # pid_net gets pulled by u_grad's contribution to the critic's TD.
        # See Eq. 14: gradient of total loss w.r.t. θ_st routes through ∂u/∂θ_st.
        # We couple by adding u_grad * (-td.detach()) so higher u under a
        # positive TD is rewarded (interpreting TD as advantage).
        pid_net_signal = -td.detach() * u_grad

        total_loss = l_actor + l_critic + pid_net_signal

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        report = StepReport(
            step=env.current_step,
            ts=next_state.name,
            action=action,
            reward=reward,
            td=float(td.detach()),
            actor_loss=float(l_actor.detach()),
            critic_loss=float(l_critic.detach()),
            predicted_vol=float(mu.detach()),
            actual_vol=next_realized_vol,
        )
        reports.append(report)
        if on_step is not None:
            on_step(report)

        prev_realized_vol = next_realized_vol
        prev_action = action

    return reports
