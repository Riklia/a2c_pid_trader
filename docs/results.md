# Results

## Key Takeaways

1. **Robustness to bad settings:** When the static PID is misconfigured (wrong $K_p$, lagging 30-day vol forecast, or a coin with different dynamics like ADA), online `A2CPID` automatically corrects the gains and significantly improves vol tracking, Sharpe, or drawdown.
2. **Edge on well-tuned multi-asset portfolios:** With anti-windup fixes and daily gain scheduling (`gain_hold = 24`, `max_position = 2.0`), `a2c_online` beats `static_pid` out-of-sample on BTC (`+20.62%` vs `+19.45%` CAGR, Sharpe `0.693` vs `0.664`), ETH (`+21.86%` vs `+19.85%` CAGR, Sharpe `0.756` vs `0.701`), and an equal-weight 3-asset portfolio (`+20.16%` vs `+19.43%` CAGR, Sharpe `0.791` vs `0.774`).
3. **Online adaptation is essential:** A network trained on past data and then frozen (`a2c_pid`) barely beats `static_pid`. Continuing to learn online while trading (`a2c_online`) is what drives the improvement.
4. **Train on volatility (QLIKE), never on hourly returns:** In a 24-point grid search, every configuration trained with return utility (`w_return = 1`) lost to `static_pid`, while every configuration trained purely on volatility tracking + costs (`w_return = 0`) beat `static_pid`.

---

## 1. Synthetic Benchmark (Known True Volatility)

Tested on 8,000-bar synthetic series (4,000 train / 4,000 test, target vol `0.25`, 5 seeds). Metric: rolling portfolio volatility RMSE (lower is better).

| Scenario | `static_pid` | `a2c_pid` (frozen) | `a2c_online` | Result (`online` vs `static`) |
|---|---|---|---|---|
| Correct design (control case) | 0.0142 | 0.0143 | **0.0137** | 4/5 seeds won (tie) |
| 2× forecast bias appears in test | 0.0639 | 0.0897 | **0.0269** | **5/5 won (2.4× lower error)** |
| Forecast bias drifts 1× → 2× | 0.0304 | 0.0484 | **0.0159** | **5/5 won (1.9× lower error)** |
| Volatility regime switching | 0.0598 | 0.0603 | **0.0591** | 3/5 won (tie) |

---

## 2. Stress Tests on Real Data (9 Conditions, 1× Max Position)

Walk-forward on hourly Binance perpetuals (1-year train, 3-month test, 3 seeds per fold, 4 bps fee + funding):

| Condition | Design flawed? | Test quarters | `a2c_online` vs `static_pid` |
|---|---|---|---|
| BTC, default design (`fc=168h, Kp=1.0`) | No | 21 | Parity |
| ETH, using BTC design | No | 20 | Parity |
| XRP, using BTC design | Unclear | 17 | Parity (95% CI includes 0) |
| BTC, `Kp = 1.4` (+40% too high) | Mildly | 21 | Parity, leaning better |
| BTC, 30-day vol forecast (`720h`, too slow) | **Yes** | 21 | **Better sizing** 17/21 ($p = 0.004$), **lower DD** 15/21 ($p = 0.04$) |
| BTC, 24h vol forecast (`24h`, too noisy) | **Yes** | 21 | **Higher return** (+0.51 pp/qtr frozen), worse sizing |
| BTC, `Kp = 0.7` (−30% too low) | **Yes** | 21 | **Higher Sharpe** 17/21 ($p = 0.004$), **return** 16/21 ($p = 0.013$), **lower DD** 15/21 ($p = 0.04$) |
| ADA, using BTC design | **Yes** | 17 | **Better sizing** 14/17 ($p = 0.006$), **lower DD** |

---

## 3. Full Walk-Forward Performance (`max_position = 2.0`, 2021–2026)

With all anti-windup and optimizer-state fixes enabled, evaluated on single-asset features (`realized_vol`, `vol_forecast`, `log_return`, `funding`, `mom_720h`, `mom_2160h`), stitched across all out-of-sample test quarters (mean over 3 seeds for A2C):

| Asset / Portfolio | Controller | Total Return | CAGR | Full Sharpe | Max DD | Turnover / qtr | Notes vs `static_pid` |
|---|---|---|---|---|---|---|---|
| **BTCUSDT** (21 qtrs, `gh=24, w_gain=0.1`) | `buy_hold` | +26.0% | +4.56% | +0.080 | 76.8% | 1.0 | — |
| | `vol_target` | +135.3% | +17.98% | +0.615 | 35.5% | 11.5 | — |
| | `static_pid` | +150.9% | +19.45% | +0.664 | 35.8% | 11.5 | Baseline |
| | `a2c_pid` (frozen) | +153.7% | +19.70% | +0.664 | 35.1% | 12.1 | 12/21 quarters won |
| | **`a2c_online`** | **+163.9%** | **+20.62%** | **+0.693** | **35.0%** | **11.9** | **13/21 Sharpe & DD, +1.17 pp/yr CAGR** |
| **BTCUSDT** (21 qtrs, `gh=24, w_gain=0.3`) | **`a2c_online`** | **+157.8%** | **+20.08%** | **+0.681** | **35.4%** | **11.7** | **14/21 Sharpe (`p=0.095`), 15/21 return (`p=0.039`)** |
| **ETHUSDT** (20 qtrs, `gh=1, w_gain=0.1`) | `buy_hold` | −21.7% | −4.85% | −0.070 | 81.1% | 1.0 | — |
| | `vol_target` | +134.3% | +18.85% | +0.664 | 34.4% | 8.8 | — |
| | `static_pid` | +144.1% | +19.85% | +0.701 | 34.1% | 8.9 | Baseline |
| | `a2c_pid` (frozen) | +160.1% | +21.40% | +0.738 | 34.9% | 10.0 | 13/20 quarters won |
| | **`a2c_online`** | **+165.0%** | **+21.86%** | **+0.756** | **34.0%** | **9.6** | **13/20 Sharpe, +2.01 pp/yr CAGR** |
| **3-Asset Equal-Weight** (BTC+ETH+BNB) | `buy_hold` | +15.0% | +2.73% | +0.045 | 76.0% | — | — |
| | `static_pid` | +151.4% | +19.43% | +0.774 | 32.3% | — | Baseline |
| | **`a2c_online`** | **+159.5%** | **+20.16%** | **+0.791** | **32.6%** | — | **Best CAGR & Sharpe overall** |

### What does `PIDNet` actually learn?

Across all 21 BTC folds, on invested bars (`pos_static > 0`), the difference between `a2c_online` and `static_pid` has a `-0.648` correlation with `pos_static` at almost identical turnover (`11.9` vs `11.5`). Because 7-day crypto volatility strongly mean-reverts, `PIDNet` learns to slightly damp effective $K_p$ below `1.0`—taking less extreme leverage in unusually quiet markets and cutting exposure less aggressively right after temporary volatility spikes.

---

## 4. Reproduce

```bash
# 1. Synthetic benchmark (~10 min on 20 cores)
python scripts/synthetic_benchmark.py --seeds 5 --workers 20

# 2. Walk-forward on BTC with 2x max position, 6 single-asset features, and daily gain hold
python scripts/run_walk_forward.py configs/walk_forward.json --workers 20 --tag btc_lev2_gh24 \
    --set env.max_position=2.0 --set a2c_pid.gain_hold=24 \
    --set 'a2c_pid.features=[{"column":"realized_vol"},{"column":"vol_forecast"},{"column":"log_return"},{"column":"funding"},{"column":"mom_720h"},{"column":"mom_2160h"}]'

# 3. Stress-test example: 30-day lagging forecast
python scripts/run_walk_forward.py configs/walk_forward.json --workers 20 --tag A_btc_fc720 \
    --set controller.forecast_vol_window='"720h"'

# 4. Compare runs against static PID with paired sign test & bootstrap CI
python scripts/compare_runs.py results/run_*

# 5. Grid search with time holdout (5 tuning folds -> 16 unseen holdout folds)
python scripts/grid_search.py configs/grid_search.json --workers 20
```
