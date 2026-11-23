# Control Formulation

We frame portfolio volatility targeting as a closed-loop control problem with a 2-degree-of-freedom (2-DOF) log-space PID controller and online neural gain scheduling.

## 1. The Plant (Market + Portfolio)

Unlike a physical robot where the controller drives the whole plant, our system has two parts in series:

1. **Market (exogenous disturbance):** Produces hourly log returns and market volatility $\sigma_k$. Our trades do not move market volatility.
2. **Portfolio (controlled output):** We control **portfolio volatility**, $\text{pv}_k = |u_k| \cdot \sigma_k$, by choosing exposure. Portfolio equity updates each bar by:

$$
E_{k+1} = E_k \left(1 + u_k (e^{r_{k+1}} - 1) - \text{fees} - \text{funding}\right)
$$

## 2. Log-Space PID Control Law

Direction (`+1` long or `0` flat, from 30-day momentum) and size are separated. Size is computed in log space (`control/vol_target.py`):

$$
\log |u_k| = K_p\, e_k + K_i\, e_{\text{int},k} + K_d\, e_{d,k}, \qquad |u_k| \le u_{\max}
$$

- **Proportional term (feedforward on market vol):**

$$
e_k = \log\left(\frac{\text{target}}{\hat\sigma_k}\right)
$$

  where $\hat\sigma_k$ is the 7-day (`168h`) realized market volatility.
  *Why log space?* When `Kp = 1` and `Ki = Kd = 0`, we get $|u_k| = \text{target}/\hat\sigma_k$, which is textbook volatility targeting. A linear PID (`u = Kp * (target - vol)`) cannot represent an inverse volatility relationship.
- **Integral term (closed-loop feedback on portfolio vol):**

$$
e_{\text{int},k} = \sum_{j < k} \log\left(\frac{\text{target}}{\text{pv}_j}\right)
$$

  where $\text{pv}_j$ is the annualized realized volatility of the controller's own last $W = 168$ **invested** (non-zero position) bars. Skipping flat bars prevents zero returns during cash periods from artificially deflating `portfolio_vol` and winding up the integral on re-entry.
- **Derivative term (rate of change of market vol):**

$$
e_{d,k} = e_k - e_{k-1}
$$

### Anti-Windup

Two mechanisms prevent saturation problems:
- **Loop anti-windup:** The integral is capped at $|e_{\text{int}}| \le \ln(c) / K_i$ (where $c = 1.5$ is the max integral multiplier) and stops accumulating positive errors whenever exposure is pinned at `max_position`.
- **Gradient anti-windup (`_AntiWindupClamp`):** Standard `torch.clamp` zeroes all gradients at `max_position`. Our custom clamp blocks outward gradients when saturated, but passes inward gradients so the network can still learn to pull leverage back down.

## 3. Analytic Static Gains (SIMC Tuning)

Static gains are derived analytically in `control/tuning.py`, not hand-tuned:

- **$K_p = 1.0$:** Gives exact baseline vol targeting $|u| = \text{target} / \hat\sigma$.
- **$K_d = 0.0$:** If $\hat\sigma$ is unbiased, no derivative action is needed at baseline; `PIDNet` learns the derivative gain if the forecast lags.
- **$K_i = 1 / W = 1 / 168 \approx 0.00595$:** In log space, $\log \text{pv} = \log |u| + \log \sigma$ has unit static gain ($k = 1$). Measuring `pv` over a rolling window of $W = 168$ bars acts like a sensor delay of $\theta = W/2 = 84$ bars. By Skogestad's SIMC rule (2003) with robust closed-loop time constant $\tau_c = \theta$:

$$
K_i = \frac{1}{k(\tau_c + \theta)} = \frac{1}{2\theta} = \frac{1}{W}
$$

  This gives a ~61° phase margin and corrects steady forecast bias over ~3.5 days.

## 4. Self-Tuning with `PIDNet`

At each bar, `PIDNet` (a 2-layer MLP, `32×32`, ~1.5k parameters) outputs bounded adjustments to the static gains:

$$
K(t) = K^{\text{static}} + \Delta K(s_t), \qquad \Delta K = \text{bounds} \cdot \tanh(\text{MLP}(s_t))
$$

- Output weights are initialized near zero, so `A2CPID` starts out identical to `StaticPID`.
- Default bounds are `[0.5, 0.003, 20.0]` for $[\Delta K_p, \Delta K_i, \Delta K_d]$ (the larger derivative bound matches the small hourly scale of $e_{d,k}$).
- `gain_hold` (e.g., `24`) holds the network's gain adjustments constant for 24 hours so daily adaptation aligns with daily optimizer steps without hourly fee churn.

## 5. Online Pathwise Learning

Because the market is exogenous, each bar's loss $L_k$ is a smooth function of our position. We backpropagate directly through the control law:

$$
\frac{\partial L_k}{\partial \theta} = \frac{\partial L_k}{\partial u_k}\,\frac{\partial u_k}{\partial \Delta K}\,\frac{\partial \Delta K}{\partial \theta}
$$

Using the QLIKE volatility-tracking loss (Patton, 2011):

$$
Q_k = \frac{(u_k R_{k+1})^2 B}{\text{target}^2} - \log(u_k^2)
$$

the expected gradient with respect to $|u_k|$ vanishes at:

$$
\frac{\partial \mathbb{E}[Q_k]}{\partial |u_k|} = 0 \implies |u_k^*| = \frac{\text{target}}{\sqrt{\mathbb{E}[R_{k+1}^2 B]}} = \frac{\text{target}}{\sigma_{\text{true}}}
$$

This trains `PIDNet` directly from realized next-bar returns without needing true volatility labels or a critic network.
