# Results: self-tuning PID vs fixed-gain PID

## Claim

A self-tuning PID (static gains plus network-adjusted gains, learning online)
**matches a fixed-gain PID when the fixed design is right, and beats it when
the design does not match the market.** It never does significantly worse on
return or drawdown. The adaptation has to continue while trading: the same
network trained once and then frozen does not show the advantage.

Profit itself comes from the control structure plus the direction signal, not
from the network: see "Baseline strategy" below.

## Setup

- **Controller**: position = direction × min(exp(Kp·e + Ki·∫e + Kd·Δe), max
  position), e = log(target vol / 7-day vol forecast). Static gains are
  derived, not tuned: Kp = 1, Ki = 1/168 (SIMC), Kd = 0. Direction: long when
  the last 30 days' return is positive, flat otherwise, decided daily. See
  [control_formulation.md](control_formulation.md).
- **Static PID**: the controller with the derived gains.
- **A2C-PID**: the same controller plus bounded gain adjustments from PIDNet,
  trained on one year by backpropagating a vol-tracking + cost loss through
  the position. Evaluated two ways on each test quarter:
  - **frozen**: weights fixed during the test quarter;
  - **online**: keeps learning bar by bar from realized returns (causal: an
    update only uses returns that already happened).
- **Data**: Binance perpetuals, hourly, fees 4 bp per unit turnover, funding
  charged. Walk-forward: 1-year train, 3-month test, sliding by 3 months;
  BTC test quarters 2021-03 .. 2026-09 (21), fewer for later-listed coins.
  3 seeds per A2C run, averaged per quarter.
- **Statistics**: per quarter, A2C (mean over seeds) minus static PID on the
  same quarter. Reported: quarters won, one-sided exact sign test, mean
  difference with a bootstrap 95% CI.

## Baseline strategy (static PID, BTC, continuous 2021-03 .. 2026-09)

| | CAGR | Sharpe | max drawdown |
|---|---|---|---|
| buy & hold | 8.4% | 0.15 | 77% |
| static PID + momentum | ~16% | ~0.58 | ~35% |

Assumption behind the profit: 30-day time-series momentum keeps working in
crypto. It lags strong bull markets (2024: +31% vs +96% for buy & hold).

## Synthetic benchmark (known ground truth)

Markets with a known true vol, so the right exposure is known exactly. The
network only sees causal vol estimates. Metric: rolling portfolio-vol RMSE vs
the 0.25 target on the test series, lower is better, mean over 5 seeds.

| scenario | static PID | A2C frozen | A2C online | online better |
|---|---|---|---|---|
| design correct (control case) | 0.0142 | 0.0143 | 0.0136 | 5/5, tiny margin |
| forecast bias 2× appears after training | 0.0639 | 0.0884 | **0.0239** | 5/5, 2.7× |
| forecast bias drifts 1× → 2× | 0.0304 | 0.0473 | **0.0166** | 5/5, 1.8× |
| vol regime switching, lagging forecast | 0.0598 | 0.0602 | 0.0596 | 3/5, tie |

## Real data (walk-forward)

Online A2C vs static PID. "Design wrong?" states whether the fixed design is
known to be off in that condition.

| condition | design wrong? | quarters | result |
|---|---|---|---|
| BTC, correct design | no | 21 | tie |
| BTC, 2× leverage allowed | no | 21 | tie |
| ETH traded with BTC design | no | 20 | tie |
| XRP traded with BTC design | unclear | 17 | tie (Sharpe −0.026, CI includes 0) |
| BTC, Kp = 1.4 (40% high) | mildly | 21 | tie, leaning better |
| BTC, 30-day vol forecast (too slow) | yes | 21 | **sizing better** 17/21, p = 0.004; **drawdown better** 15/21, p = 0.04 |
| BTC, 24-hour vol forecast (too noisy) | yes | 21 | return better (frozen: +0.51 pp/quarter, CI [+0.08, +0.97]); sizing worse |
| BTC, Kp = 0.7 (30% low) | yes | 21 | **Sharpe better** 17/21, p = 0.004; **return** 16/21, p = 0.013; **drawdown** 15/21, p = 0.04 |
| ADA traded with BTC design | yes | 17 | **sizing better** 14/17, p = 0.006; drawdown better, CI excludes 0 |

"Sizing" is |realized vol while invested − target|; flat periods are excluded
because the momentum rule decides those, not the PID.

## Hyperparameter search (BTC, correct design)

A grid search over A2C-PID's training hyperparameters, with a time holdout
so the reported numbers are not the ones used for selection:

- **Tuning**: each grid point is trained and evaluated online on the first 5
  BTC quarters only (2021-03 .. 2022-06, 1 seed), scored by its mean paired
  Sharpe gain over static PID.
- **Holdout**: only the winner then runs, with 3 seeds, on the 16 later
  quarters (2022-06 .. 2026-09), unseen during selection.

Grid (24 points): learning rate {0.003, 0.01, 0.03} × gain regularization
`w_gain` {0.03, 0.3} × return term `w_return` {0, 1} × gain bounds
{(0.5, 0.003, 20), 2× that}. Everything else as in the real-data runs.

**Tuning quarters**, online A2C vs static PID, mean Sharpe gain:

| `w_return` | points | mean Sharpe gain, range |
|---|---|---|
| 0 (vol tracking + costs only) | 12 | all positive, +0.001 .. +0.046 |
| 1 (adds the return term) | 12 | all negative, −0.004 .. −0.226 |

Top of the ranking (the differences between them are within noise: the
winner won 3 of 5 quarters):

| lr | `w_gain` | gain bounds | mean Sharpe gain | quarters won |
|---|---|---|---|---|
| **0.01** | **0.3** | **2×** | **+0.046** | 3/5 |
| 0.03 | 0.3 | 2× | +0.036 | 2/5 |
| 0.003 | 0.03 | 1× | +0.036 | 4/5 |
| 0.03 | 0.03 | 1× | +0.034 | 2/5 |
| 0.01 | 0.03 | 2× | +0.031 | 3/5 |
| 0.01 | 0.1 | 1× (current default, not in the grid; seed 0 of the BTC run) | +0.017 | 3/5 |

**Holdout quarters** (16), online A2C vs static PID:

| | default hyperparameters | tuned (winner) |
|---|---|---|
| Sharpe | 10/16 won, +0.010 | 11/16 won, +0.021 (p = 0.11, CI [−0.013, +0.059]) |
| return per quarter | +0.13 pp | +0.25 pp (CI includes 0) |
| max drawdown | ≈0 | +0.06 pp (CI includes 0) |
| sizing error while invested | slightly worse (CI excludes 0) | ≈0 (CI includes 0) |

Frozen A2C with the tuned hyperparameters: Sharpe −0.005 vs static PID.

Takeaways:

- **Train the network on vol tracking, not on returns.** Adding the
  mean-variance return term made A2C worse than static PID in all 12 grid
  points that had it. Hourly returns are too noisy to learn from in a year.
- Tuning roughly doubles A2C's edge on unseen quarters and removes its small
  sizing penalty, but in this correct-design setting the gain stays within
  noise, as expected; the significant results remain the design-error
  conditions above.
- Online adaptation is still what matters: tuned but frozen, A2C does not
  beat static PID.
- The defaults in `configs/walk_forward.json` were left unchanged so the
  tables above stay reproducible.

## Caveats

- Effects are modest in size (Sharpe +0.03 per quarter where significant).
  The claim is robustness to design error, not a higher return engine.
- Sign tests are per condition and not corrected for testing 9 conditions;
  the strongest results (p ≈ 0.004–0.006) survive a Bonferroni correction
  for 9 tests at the 5% level only marginally (threshold 0.0056).
- Synthetic scenarios were designed knowing the mechanism. They show *when*
  adaptation helps; the real-data runs are the evidence that it happens.
- Static PID's integral term can correct exposure by at most 1.5× (its
  anti-windup limit), less than the 2× synthetic bias needs. A looser limit
  would narrow the synthetic gap.
- Momentum direction is the profit assumption; all comparisons share it.

## Reproduce

```bash
# synthetic benchmark (~10 min on 20 cores)
python scripts/synthetic_benchmark.py --seeds 5 --workers 20

# one real-data condition (~15-20 min on 20 cores), e.g. slow forecast
python scripts/run_walk_forward.py configs/walk_forward.json --workers 20 --tag A_btc_fc720 \
    --set controller.forecast_vol_window='"720h"'

# paired statistics for any set of runs
python scripts/compare_runs.py results/run_*_A_*

# hyperparameter search with time holdout (~50 min on 20 cores), grid in configs/grid_search.json
python scripts/grid_search.py configs/grid_search.json --workers 20
```

The real-data conditions above used the traded symbol's own features
(`realized_vol`, `vol_forecast`, `log_return`, `funding`, `mom_720h`,
`mom_2160h`) via `--set a2c_pid.features=...`, and varied one setting each:
`controller.forecast_vol_window` (`720h`, `24h`), `env.max_position` (`2.0`),
`static_pid.kp` (`0.7`, `1.4`), `data.symbol` (`ETHUSDT`, `ADAUSD`, `XRPUSD`).
