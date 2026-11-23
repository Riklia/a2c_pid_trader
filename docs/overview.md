# Overview

## What this project does

**Volatility targeting** scales position size inversely with market volatility so portfolio risk stays steady: hold more when the market is calm, and hold less when it is turbulent.

In crypto, a fixed-gain PID controller struggles when volatility regimes shift or when the volatility forecast lags. This project keeps the PID formula as the policy, but adds a small neural network (`PIDNet`) that nudges the PID gains $(K_p, K_i, K_d)$ online from recent market state.

> **Why is it called `a2c_pid`?** The project started as an Actor-Critic implementation following Sharifi & Alasty (2022). Because market returns are exogenous (our trades do not move the market), each bar's loss is directly differentiable with respect to position size. We replaced the noisy Actor-Critic TD update with exact pathwise gradients through the control law, while keeping the `A2CPID` class and config names for compatibility.

## How a single bar works

Position direction and position size are decided separately:

$$
u_k = \text{direction}_k \times \min\left(\exp\big(K_p e_k + K_i e_{\text{int},k} + K_d e_{d,k}\big),\ u_{\max}\right)
$$

1. **Direction (`direction_k`):** Decided by 30-day time-series momentum (`mom_720h`) by default. If the last 30 days' return is positive, direction is `+1` (long); otherwise `0` (flat in cash). Updated once per day (`direction_hold = 24`) to avoid whipsaw fees.
2. **Base size (`StaticPID`):** Computed in log space from three error signals:
   - $e_k = \log(\text{target} / \hat\sigma_k)$ compares target volatility (35%) to the 7-day market volatility forecast (when `Kp = 1` and `Ki = Kd = 0`, size is simply `target / vol_forecast`).
   - $e_{\text{int},k}$ sums past tracking errors using the portfolio's own realized volatility (`pv`) over the last 168 *invested* (non-flat) bars.
   - $e_{d,k} = e_k - e_{k-1}$ measures how fast the volatility forecast is changing.
3. **Adaptive gain nudge (`PIDNet`):** A small MLP (`32×32`) reads recent positions, the three PID errors, and z-scored market features, and outputs bounded adjustments $(\Delta K_p, \Delta K_i, \Delta K_d)$ added to the static gains. Updates can be held for `gain_hold` bars (e.g., 24h) to keep turnover low.
4. **Execution (`TradingEnv`):** The position $u_k$ is applied to the *next* bar's simple return, minus a 4 bps fee on position changes and 8-hour perpetual funding payments.

## How training works

Because the market is exogenous, we don't need a critic network or replay buffer. The controller runs online through the data, bar by bar, and accumulates exact gradients over `update_every` (24) bars before taking an Adam step:

$$
L_k = w_{\text{track}}\,Q_k + C_k + w_{\text{gain}}\,\lVert \Delta K / K_{\text{bound}} \rVert^2 - w_{\text{return}}\,U_k
$$

- **Volatility tracking ($Q_k$):** Quasi-likelihood (QLIKE) loss on next bar's realized portfolio variance:

$$
Q_k = \frac{(u_k r_{k+1})^2 N}{\text{target}^2} - \log(u_k^2)
$$

  Its minimum in expectation is exactly $|u_k| = \text{target} / \sigma_{\text{true}}$.
- **Trading costs ($C_k$):** Exact turnover fee plus funding payment.
- **Gain penalty ($w_{\text{gain}}$):** L2 penalty that keeps $\Delta K$ near zero unless adaptation clearly helps.
- **Return utility ($U_k$, disabled by default):** Hourly returns are ~30× noisier than volatility, so training on returns (`w_return > 0`) hurts out-of-sample performance. Default is `w_return = 0`.

## How a walk-forward experiment runs

1. Load 1-minute candles and funding rates from Postgres, resample to 1-hour bars, and compute features (cached in `.cache/features/`). Bars before funding history starts are kept with `funding = 0`.
2. Split history into sliding windows: **1 year (8760 bars) of training** followed by **3 months (2160 bars) of testing**, stepping forward by 3 months (21 test folds on BTC from 2021 to 2026).
3. On each fold, evaluate five controllers on the test quarter:
   - `buy_hold` — constant `1.0` position.
   - `vol_target` — proportional sizing only ($K_p=1, K_i=K_d=0$) with the same momentum direction filter.
   - `static_pid` — fixed analytically derived gains ($K_p=1, K_i=1/168, K_d=0$).
   - `a2c_pid` (frozen) — trained on the 1-year train window, weights frozen during the test quarter.
   - `a2c_online` — trained on the 1-year train window, and continues updating weights every 24 bars during the test quarter (using only already-realized returns).
4. Save config, per-fold metrics (`metrics.csv`), and bar-by-bar traces (`traces/*.parquet`) into `results/run_<timestamp>/`.

## No look-ahead guarantees

- Every feature at bar $k$ uses only data up to bar $k$'s close.
- Position $u_k$ is applied strictly to bar $k+1$'s return.
- Online updates in `a2c_online` only use realized returns from completed bars.
- Unit tests verify bit-identical feature tables when truncating future data.
