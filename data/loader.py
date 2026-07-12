from dataclasses import dataclass
from sqlalchemy import create_engine, text
import pandas as pd

@dataclass
class DBConfig:
    dsn: str
    schema: str = "binance"

def get_engine(cfg: DBConfig):
    return create_engine(cfg.dsn)

def load_trades(engine, symbol: str = "BTCUSDT", quarter: int | None = None,
                 start_ts: int | None = None, end_ts: int | None = None,
                 schema: str = "binance") -> pd.DataFrame:
    conds = ["symbol = :symbol"]
    params = {"symbol": symbol}
    if quarter is None:
        conds.append("quarter IS NULL")
    else:
        conds.append("quarter = :quarter")
        params["quarter"] = quarter
    if start_ts is not None:
        conds.append("ts >= :start_ts"); params["start_ts"] = start_ts
    if end_ts is not None:
        conds.append("ts <= :end_ts"); params["end_ts"] = end_ts

    query = f"""
        SELECT ts, symbol, open, close, low, high, volume, trades, quarter
        FROM {schema}.bnc_trade
        WHERE {' AND '.join(conds)}
        ORDER BY ts
    """
    df = pd.read_sql(text(query), engine, params=params)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df.set_index("ts")

def load_funding(engine, symbol: str = "BTCUSDT",
                  start_ts: int | None = None, end_ts: int | None = None,
                  schema: str = "binance") -> pd.DataFrame:
    conds = ["symbol = :symbol"]
    params = {"symbol": symbol}
    if start_ts is not None:
        conds.append("ts >= :start_ts"); params["start_ts"] = start_ts
    if end_ts is not None:
        conds.append("ts <= :end_ts"); params["end_ts"] = end_ts

    query = f"""
        SELECT ts, symbol, mark_price, index_price, estimated_settle_price,
               interest, funding, time_left_seconds
        FROM {schema}.bnc_funding
        WHERE {' AND '.join(conds)}
        ORDER BY ts
    """
    df = pd.read_sql(text(query), engine, params=params)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df.set_index("ts")