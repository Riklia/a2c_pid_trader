# Overview

## What problem the project solves

Volatility targeting is a standard risk-control technique: pick a target
portfolio volatility and size each position so that the portfolio hits 
that target. When the market is calm, hold more. When the market is 
turbulent, hold less.

Doing this with a fixed-gain PID does not work well on crypto, because
crypto has long regimes with very different realized volatility. A set of
gains tuned on 2023 behaves badly in 2024, and vice versa. This project
asks: can a neural network tune the PID gains online, bar by bar, so the
same controller works across regimes?

## How it works, end to end

A single walk-forward experiment does this:

1. Pull BTCUSDT minute candles and funding snapshots from Postgres.
2. Resample to hourly, align funding (every 8 hours), compute realized
   volatility and the three PID error signals from a target vol setpoint.
3. Slice the resulting series into fold pairs: one year of training
   followed by three months of test, sliding forward.
4. For each fold:
   - Freshly initialize an A2C-PID controller.
   - Run it online through the training window, one bar per step, one
     optimizer step per bar, with networks learning in place.
   - Freeze the networks and run the controller through the test window.
     Record positions, equity, rewards.
   - Separately run a fixed-gain static PID on the same test window as
     a baseline.
5. Write everything into a timestamped results directory.

## How a single step works

At each bar, A2C-PID does:

- Read the current state (realized vol, PID errors, funding, position
  history).
- Feed a normalized version into a small MLP that outputs three dynamic
  gain adjustments.
- Add those to the static gains, compute $u = Kp \cdot e_p + Ki \cdot e_i + Kd \cdot e_d$,
  clip to [-1, 1]. This is the position for the next bar.
- During training only: an actor network predicts the next realized vol,
  a critic estimates value, and the TD error is used to update all three
  networks. The gradient flows into the PID network through the fact that
  `u` depends on its outputs.

The environment then applies that position against the next bar's return,
charges a taker fee on turnover, and charges funding on the settlement
bar if there is one.

## What makes this not just a trading bot

A few deliberate choices from the paper carried over:

- **The policy is a PID, not a neural network.** The MLP outputs gain
  deltas, not actions. This keeps the controller interpretable and gives
  it an inductive bias that pure RL lacks.
- **Actor-critic is scaffolding, not the policy.** The actor predicts
  next-state volatility and acts as a learned model, not a policy head.
  The critic estimates V. Both exist to provide a learning signal to the
  PID network.
- **Online, no replay buffer.** Each bar updates the networks immediately
  with the data from that one bar. This matches the paper's quadrotor
  setup and makes the system naturally adaptive at test time.

## Reward function

Let $e_k = r - |u_k|\sigma_k$ be the tracking error at step $k$, where $r$
is the target volatility, $u_k$ is the position, and $\sigma_k$ is the
realized market volatility. The reward is

$$
R_{k+1} = -w_\text{err}\,e_k^2 - w_\text{rate}\,(e_k - e_{k-1})^2 - w_\text{act}\,u_k^2.
$$

The quadratic error term is the control-theoretic tracking cost. The
action term penalizes large positions, loosely corresponding to
transaction cost. Weights are hyperparameters in the config.

Note the controlled variable is `|position| · realized_vol`, not just
realized vol. The market's volatility is not something the controller
can move. What it can move is the portfolio's volatility, by scaling
exposure. This is the correct framing for the task and differs from the
quadrotor paper, where the plant is fully controller-driven.

## Guarantees against look-ahead

Three places where the pipeline could leak future information, each
blocked:

- Funding alignment uses only funding snapshots up to each bar's close.
- The environment applies a chosen position to the *next* bar's return,
  never the current bar.
- A regression test rebuilds features from a truncated prefix and asserts
  that values up to the cutoff are bit-identical to the full version.

## What goes into a results directory

Every walk-forward invocation creates a new `results/run_<timestamp>/`.
Nothing is overwritten. The directory contains:

- The exact config that produced the run.
- CSV with metrics: one row per (fold, controller) with return, Sharpe,
  drawdown, vol tracking RMSE.
- Per-fold raw traces: equity, position, reward, action at each bar.
- Model checkpoints: trained A2C-PID weights per fold.

This is enough to re-plot, re-evaluate, or diff two experiments without
re-running training.