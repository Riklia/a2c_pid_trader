# Control-theoretic formulation

The project is a closed-loop control problem with a stochastic plant, cast
so it matches the actor-critic PID from Sharifi & Alasty (2022).

## The loop

The controller sees three precomputed error signals ($e_p$, $e_i$, $e_d$)
derived from $r − σ_realized$. These are features, not the real tracking
error. The real tracking error depends on $u$ and only enters through the
reward.

## The plant

Two things composed in series:

1. The market, which produces $σ_realized$ and $log_return$
   exogenously. The controller cannot influence it.
2. **Portfolio dynamics**, where $y = |u| · σ_realized$ and equity updates
   by $(1 + u · \text{log_return} − \text{fees} − \text{funding})$ per bar.

This differs from the quadcopter case, where the plant is fully controlled
by `u`. Here, half of the plant is noise the controller must respond to.

## The cost function

Reward is quadratic in tracking error and action size:

$$
R_{k+1} = -w_\text{err}\,(r - |u_k|\sigma_k)^2 \;-\; w_\text{rate}\,\Delta e_k^2 \;-\; w_\text{act}\,u_k^2
$$

Training maximizes $Σ γ^k R_{k+1}$, which is equivalent to minimizing an
LQR-style discounted cost. The weights trade off tracking tightness,
smoothness of the controlled variable, and position magnitude (loosely a
transaction-cost proxy). All three are hyperparameters in the training
config.

## Why the controller is a PID and not just the network

Classical PID: $u = Kp \cdot e_p + Ki \cdot e_i + Kd \cdot e_d$, gains fixed. Works
poorly when the plant's statistics shift between training and deployment.

Self-tuning PID keeps the structure but makes the gains the sum of a
static part and a dynamic part:

$$
K_p(t) = K_p^\text{static} + K_p^\text{dyn}(s_t)
$$

The static part is set once (Ziegler-Nichols or by hand). The dynamic
part is a neural network output that reacts to the current state. The
policy stays interpretable, three gains and a sum, but it adapts.

## Where actor-critic fits

Actor and critic are not the policy. The policy is the PID formula above.

- **Critic** estimates V(state). It gives a TD error after each step.
- **Actor** predicts the next realized volatility as a Gaussian. This is
  a learned model of the plant, not a policy head. Its sigma output keeps
  the system exploratory.
- **PIDNet** (the dynamic-gains network) is updated by a gradient of the
  form $−TD \cdot u$, so actions with positive advantage get reinforced
  through their effect on $u$.

Together: critic grades the controller's actions, actor keeps a
non-degenerate learning signal, PIDNet is the thing that actually changes.

## Walk-forward evaluation

Training is online and incremental. A single train/test split would
discard most of the data and depend heavily on which window was picked.
Walk-forward slides a 1-year train window followed by a 3-month test
window across the full history. Each fold gets a fresh controller.
Metrics are reported per fold and averaged, so no single market regime
dominates the result.

The static PID baseline runs on the same test windows without any
training, since its gains do not change.