# a2c_pid_trader

A self-tuning PID controller that sizes crypto positions so that portfolio
volatility tracks a configured target. Adapts the actor-critic architecture
from Sharifi & Alasty, [Self-Tuning PID Control via a Hybrid
Actor-Critic-Based Neural Structure for Quadcopter Control][paper] (2022),
originally for quadcopter attitude control, to a financial setting.

[paper]: https://arxiv.org/abs/2307.01312

The core idea: a classical PID has three fixed gains. A classical PID on
crypto cannot stay optimal across different volatility regimes. This project
keeps the PID structure but lets a neural network nudge the gains online in
response to recent market state. Actor and critic are the learning
machinery, not the policy. The policy is still a PID.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in DEV_DB_DSN
python scripts/run_walk_forward.py configs/walk_forward.json
python scripts/plot_run.py results/run_<timestamp>
```

For stepping through the environment manually:

```bash
streamlit run tools/env_viewer/app.py
```

## More

- [docs/overview.md](docs/overview.md): what the system does and how it's wired together
- [docs/control_formulation.md](docs/control_formulation.md): the control-theory framing with the math
- [docs/debugging.md](docs/debugging.md): tools and workflows when something looks wrong