"""Interactive debugger for TradingEnv on DB-loaded data.

Run from the repo root:
    streamlit run tools/env_viewer/app.py

Requires DEV_DB_DSN environment variable to be set.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import pandas as pd
import streamlit as st

from data import loader, features
from environment.trading_env import TradingEnv


STATE_COLUMNS = ["close", "log_return", "funding", "is_funding_settlement",
                  "realized_vol", "e_p", "e_i", "e_d"]


def _get_dsn() -> str:
    dsn = os.environ.get("DEV_DB_DSN")
    if not dsn:
        st.error("DEV_DB_DSN is not set. Export it or add it to a .env file in the repo root.")
        st.stop()
    return dsn


@st.cache_resource
def get_engine(dsn: str):
    return loader.get_engine(loader.DBConfig(dsn=dsn))


@st.cache_data(show_spinner="Loading data from Postgres...")
def load_feature_table(
    _engine,
    symbol: str,
    start_ts: int,
    end_ts: int,
    target_vol: float,
    freq: str,
    vol_window: str,
    integral_mode: str,
    integral_window: str,
) -> pd.DataFrame:
    trades = loader.load_trades(_engine, symbol=symbol, start_ts=start_ts, end_ts=end_ts)
    funding = loader.load_funding(_engine, symbol=symbol, start_ts=start_ts, end_ts=end_ts)
    return features.build_feature_table(
        trades, funding,
        target_vol=target_vol, freq=freq,
        vol_window=vol_window, integral_mode=integral_mode, integral_window=integral_window,
    )


def _init_env(data: pd.DataFrame, fee_rate: float):
    st.session_state.env = TradingEnv(data, fee_rate=fee_rate)
    st.session_state.trace = []


def _step_env(target_position: float, n: int = 1):
    env = st.session_state.env
    for _ in range(n):
        if env.done:
            break

        idx_before = env._step_idx
        state_before = env.data.iloc[idx_before]
        equity_before = env.equity
        position_before = env.position

        _, reward, _ = env.step(target_position)

        state_after = env.data.iloc[env._step_idx]

        st.session_state.trace.append({
            "step": len(st.session_state.trace),
            "ts_before": state_before.name,
            "ts_after": state_after.name,
            "action": target_position,
            "position_before": position_before,
            "position_after": env.position,
            "turnover": abs(target_position - position_before),
            "equity_before": equity_before,
            "equity_after": env.equity,
            "reward": reward,
            "next_bar_log_return": state_after["log_return"],
            "next_bar_funding": state_after["funding"],
            "next_bar_is_settlement": state_after["is_funding_settlement"],
        })


def main():
    st.set_page_config(page_title="TradingEnv debugger", layout="wide")
    st.title("TradingEnv debugger")
    st.caption("Inspect exact state, action, and reward at each step. No smoothing, no plots -- just numbers.")

    dsn = _get_dsn()
    engine = get_engine(dsn)

    with st.sidebar:
        st.header("Data")
        symbol = st.text_input("Symbol", value="BTCUSDT")
        start_date = st.date_input("Start (UTC)", value=datetime(2024, 1, 1).date())
        end_date = st.date_input("End (UTC)", value=datetime(2024, 1, 15).date())
        freq = st.selectbox("Bar frequency", ["1h", "4h", "1d"], index=0)

        st.header("Features")
        target_vol = st.number_input("Target annualized vol", value=0.60, step=0.05, format="%.2f")
        vol_window = st.text_input("Realized vol window", value="24h")
        integral_mode = st.selectbox("Integral mode", ["rolling", "expanding"], index=0)
        integral_window = st.text_input("Integral window", value="720h",
                                          disabled=(integral_mode == "expanding"))

        st.header("Env")
        fee_rate = st.number_input("Taker fee (fraction)", value=0.0004, step=0.0001, format="%.4f")

        load_clicked = st.button("Load & reset env", type="primary", use_container_width=True)

    if load_clicked or "env" not in st.session_state:
        start_ts = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        try:
            data = load_feature_table(
                engine, symbol, start_ts, end_ts,
                target_vol, freq, vol_window, integral_mode, integral_window,
            )
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")
            st.stop()

        if len(data) < 2:
            st.warning("Feature table is too short to step through.")
            st.stop()

        _init_env(data, fee_rate)

    env = st.session_state.env

    st.subheader("Action")
    action_col, step_col, n_col, run_col, reset_col = st.columns([2, 1, 1, 1, 1])
    with action_col:
        target_position = st.slider(
            "Target position", -1.0, 1.0, 0.0, 0.05,
            help="Fraction of equity in the position, -1 short, 1 long.",
        )
    with step_col:
        step_1 = st.button("Step", use_container_width=True)
    with n_col:
        n_steps = st.number_input("N", min_value=1, max_value=1000, value=10, label_visibility="collapsed")
    with run_col:
        step_n = st.button("Step \u00d7N", use_container_width=True)
    with reset_col:
        if st.button("Reset", use_container_width=True):
            _init_env(env.data, fee_rate)
            env = st.session_state.env

    if step_1:
        _step_env(target_position, n=1)
    if step_n:
        _step_env(target_position, n=int(n_steps))

    st.divider()

    summary_cols = st.columns(4)
    summary_cols[0].metric("Bar", f"{env._step_idx} / {len(env.data) - 1}")
    summary_cols[1].metric("Equity", f"{env.equity:.6f}")
    summary_cols[2].metric("Position", f"{env.position:+.3f}")
    last_reward = st.session_state.trace[-1]["reward"] if st.session_state.trace else None
    summary_cols[3].metric("Last reward", f"{last_reward:+.6f}" if last_reward is not None else "\u2014")

    st.subheader("State")
    st.caption("The state row at the current bar. This is what a controller would receive as input at this step.")
    current_state = env.data.iloc[env._step_idx][STATE_COLUMNS]
    state_df = current_state.to_frame("value").reset_index().rename(columns={"index": "field"})

    st.dataframe(
        state_df,
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "field": st.column_config.TextColumn("Field", width="small"),
            "value": st.column_config.TextColumn("Value", width="large"),
        },
    )

    st.subheader("Step trace")
    st.caption("One row per step taken. reward = log(equity_after / equity_before). "
                "next_bar_* columns show which incoming bar was used to compute reward.")
    if st.session_state.trace:
        trace_df = pd.DataFrame(st.session_state.trace)
        st.dataframe(trace_df, use_container_width=True, height=400)
    else:
        st.info("No steps taken yet. Move the slider and press Step.")


if __name__ == "__main__":
    main()