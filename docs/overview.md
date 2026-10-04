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
   volatility (24h), the 7-day volatility forecast used for sizing and the
   momentum signals used for direction. Funding history starts years after
   price history; bars before the first funding snapshot are kept with zero
   funding instead of being dropped.
3. Slice the resulting series into fold pairs: one year of training
   followed by three months of test, sliding forward.
4. For each fold:
   - Run three baselines on the test window: buy & hold, plain vol
     targeting, and the static PID.
   - For each seed, freshly initialize an A2C-PID controller on top of
     the *same* static PID, run it online through the training window
     with the network learning in place, then freeze it and run it
     through the test window.
5. Write everything into a timestamped results directory.

## How a single step works

Direction and size are decided separately:

$$
\text{position} = \text{direction} \times \min\!\left(e^{K_p e + K_i e_\text{int} + K_d e_d},\ \text{max\_position}\right)
$$

- **Direction** comes from 30-day time-series momentum by default
  (`trend_long_flat` on `mom_720h`): long when the last 30 days' log
  return is positive, flat otherwise, re-decided once a day
  (`direction_hold`) to avoid whipsaw fees. Other modes: long only, a
  30-day moving-average filter (`trend`), long/short. Vol targeting
  manages risk, it does not predict direction, so profit has to come from
  here. Without a direction signal every controller is mostly BTC beta: it
  wins exactly when BTC rises.
- **Size** is a PID in log space on $e = \log(\text{target}/\hat\sigma)$.
  With $K_p = 1$, $K_i = K_d = 0$ it is exactly textbook vol targeting,
  $\text{target}/\hat\sigma$. The integral term accumulates
  $\log(\text{target}/\text{portfolio vol})$, measured from the
  controller's own realized returns, so it closes the loop on what the
  portfolio actually experienced.

At each bar, A2C-PID additionally feeds its last two positions, the
law's three errors and the model features listed in `a2c_pid.features`
(which can include other symbols' data, see [features.md](features.md))
into a small MLP that outputs bounded gain adjustments added to the
static gains.

The environment then applies that position against the next bar's
(simple) return, charges a taker fee on turnover, and charges funding on
the settlement bar if there is one.

## What makes this not just a trading bot

- **The policy is a PID, not a neural network.** The MLP outputs gain
  deltas, not actions. With an untrained network the controller is
  exactly its static PID, and the learned gains stay interpretable
  ($K_p < 1$: react less to vol spikes, $K_p > 1$: react more).
- **Online, no replay buffer.** The network trains in one pass over the
  training window, bar by bar, accumulating gradients over `update_every`
  bars per optimizer step.

## Training objective

The market does not react to the position, so the loss of each bar is a
known, differentiable function of the position, and the gradient can be
taken straight through it into the network (no critic needed):

$$
L_k = -w_\text{return}\,U_k + C_k + w_\text{track}\,Q_k + w_\text{gain}\,\lVert \Delta K / K_\text{bound} \rVert^2
$$

- $U_k = u_k r_{k+1} - \tfrac{\lambda}{2}(u_k r_{k+1})^2$: mean-variance
  utility of the bar's return.
- $C_k$: fee on turnover plus funding, mirroring the environment.
- $Q_k = (u_k r_{k+1})^2 N / \text{target}^2 - \log u_k^2$: QLIKE tracking
  loss, minimized in expectation exactly at $|u| = \text{target}/\sigma$.

Single-bar returns are almost pure noise, while volatility is predictable,
so the tracking term carries far more learnable signal than the utility.
The default config therefore trains on tracking plus costs
($w_\text{return} = 0$).

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
- CSV with metrics: one row per (fold, controller, seed) with return,
  Sharpe, drawdown, realized vol, rolling vol tracking RMSE, turnover,
  and beta/alpha against BTC. Beta is the part of the return that came from
  simply being exposed to BTC; alpha (annualized) is what is left. A
  long-only controller gains whenever BTC rises, so judge strategies by
  alpha and Sharpe, not by total return in an up quarter.
- Per-fold raw traces: equity, position, reward, action at each bar.

This is enough to re-plot, re-evaluate, or diff two experiments without
re-running training.