# Control-theoretic formulation

The project is a closed-loop control problem with a stochastic plant, cast
so it stays close to the actor-critic-tuned PID from Sharifi & Alasty (2022).

## The loop

The controlled variable is **portfolio volatility**, $|u| \cdot \sigma$. The
market's own volatility $\sigma$ is exogenous; the only thing the controller
can move is exposure $|u|$.

Direction and size are separated. Direction comes from a rule (long only by
default), size from the PID.

## The plant

Two things composed in series:

1. The market, which produces $\sigma$ and the log return exogenously.
   The controller cannot influence it.
2. **Portfolio dynamics**: equity updates by
   $(1 + u \cdot (e^{\text{log\_return}} - 1) - \text{fees} - \text{funding})$
   per bar, and realized portfolio vol is measured from those returns.

## The controller

A PID in log space:

$$
\log |u_k| = K_p\, e_k + K_i\, e_{\text{int},k} + K_d\, e_{d,k},
\qquad |u_k| \le \text{max\_position}
$$

- $e_k = \log(\text{target} / \hat\sigma_k)$, where $\hat\sigma_k$ is the 7-day
  realized vol (the measured disturbance).
- $e_{\text{int},k} = \sum_{j<k} \log(\text{target} / \text{pv}_j)$, where
  $\text{pv}_j$ is the realized vol of the controller's own last 168 **invested**
  (non-zero position) returns. Flat bars are skipped so flat stretches neither
  dilute `portfolio_vol` with zeros nor wind up the integral upon re-entry.
  This is the feedback path: it depends on past actions.
- $e_{d,k} = e_k - e_{k-1}$.

Why log space: with $K_p = 1$ and the other gains at zero,
$|u| = \text{target}/\hat\sigma$, the textbook vol-targeting rule. In linear
space ($u = K_p(\text{target} - \sigma)$) no choice of gains reproduces it,
which is why the original formulation held about 0.2x exposure and never got
near its target.

Anti-windup works on both the loop and the backward pass: conditional
integration stops accumulating positive errors when exposure is pinned at
`max_position`, and training uses an inward-gradient clamp (`_AntiWindupClamp`)
that blocks outward gradients when already saturated while passing inward
gradients that pull exposure back inside bounds.

## Static gains

The static gains are derived, not tuned (`control/tuning.py`); the config
says `"derived"` and a number overrides it. In log space the loop is

- plant: $\log \text{pv} = \log|u| + \log\sigma$, a static gain $k = 1$,
- sensor: pv is the std of the last $W$ = `port_vol_window` returns. A
  step in exposure enters the window one bar at a time, which is
  approximated by a dead time $\theta = W/2$.

**$K_p = 1$.** Exact vol targeting means $|u|\sigma = \text{target}$, i.e.
$\log|u| = 1 \cdot \log(\text{target}/\sigma)$.

**$K_d = 0$.** With an unbiased forecast the proportional term already sets
the right exposure; a derivative term only helps if the forecast lags
systematically, which is left to the network to learn.

**$K_i = 1/W$.** SIMC rule (Skogestad, 2003) for an integral controller on
$k e^{-\theta s}$: $K_i = 1/(k(\tau_c + \theta))$ with the robust default
$\tau_c = \theta$ (about 60° phase margin), giving $K_i = 1/(2\theta) = 1/W$.
For $W = 168$: $K_i \approx 0.006$ per bar, a closed-loop time constant of
about 3.5 days.

**Integral limit.** The integral term adds $K_i e_\text{int}$ to log
exposure. Allowing it to change exposure by at most a factor $c$ =
`max_integral_correction` gives $|e_\text{int}| \le \ln c / K_i$, about 68
bars for $c = 1.5$.

What remains a choice, not a derivation: the target vol (a risk budget,
which should stay below typical market vol so exposure is not pinned at
`max_position`), the measurement window $W$, and $c$.

## Self-tuning

$$
K(t) = K^\text{static} + \Delta K(s_t), \qquad \Delta K = \text{bounds} \cdot \tanh(\text{MLP}(s_t))
$$

The static gains are the baseline controller. The network output starts near
zero, so A2C-PID starts out identical to it, and per-gain bounds keep the
adaptation local.

`gain_bounds` can optionally take a 4th value `[kp, ki, kd, bias]`, which adds
a bounded feedforward log-exposure offset $\Delta b(s_t)$ directly inside the
exponent. `a2c_pid.gain_hold` (e.g. `24`, matching `direction_hold` and
`update_every`) holds the network's gain adjustments across `gain_hold` bars so
online adaptation aligns with daily optimizer steps without hourly fee chatter.

## Learning signal

The paper trains the gain network through a critic's TD error. Here that
is unnecessary: the market is exogenous, so each bar's loss is a known,
differentiable function of $u$, and the exact gradient is

$$
\frac{\partial L_k}{\partial \theta} = \frac{\partial L_k}{\partial u_k}\,\frac{\partial u_k}{\partial \Delta K}\,\frac{\partial \Delta K}{\partial \theta}.
$$

The previous TD-based update ($-\text{TD} \cdot u$) did not use
$\partial L / \partial u$ at all and pushed the signed position even though the
reward depended only on $|u|$, so it carried no consistent learning signal.
The actor and critic modules are still in `models/` but are no longer used.

See [overview.md](overview.md) for the loss terms.

## Walk-forward evaluation

Training is online and incremental. Walk-forward slides a 1-year train
window followed by a 3-month test window across the full history. Each fold
gets fresh controllers, and A2C-PID is trained once per seed. Metrics are
reported per fold and averaged, so no single market regime dominates the
result. Buy & hold and plain vol targeting run alongside the static PID, so
it is visible how much of any result comes from simply holding BTC.
