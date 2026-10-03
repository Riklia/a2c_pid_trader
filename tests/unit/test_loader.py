import pandas as pd
from data import loader
import pytest

class FakeEngine:
    pass

def test_load_trades_converts_ts_to_datetime(monkeypatch):
    fake_df = pd.DataFrame({
        "ts": [1700000000, 1700000060],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "open": [60000.0, 60010.0],
        "close": [60010.0, 60005.0],
        "low": [59990.0, 60000.0],
        "high": [60020.0, 60015.0],
        "volume": [1.2, 0.8],
        "trades": [100, 90],
        "quarter": [None, None],
    })

    def fake_read_sql(query, engine, params=None):
        assert params["symbol"] == "BTCUSDT"
        return fake_df.copy()

    monkeypatch.setattr(loader.pd, "read_sql", fake_read_sql)

    result = loader.load_trades(engine=FakeEngine(), symbol="BTCUSDT")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index[0] == pd.Timestamp("2023-11-14 22:13:20")
    assert list(result.columns) == ["symbol", "open", "close", "low", "high", "volume", "trades", "quarter"]

def test_load_trades_defaults_to_quarter_is_null(monkeypatch):
    captured_query = {}

    def fake_read_sql(query, engine, params=None):
        captured_query["sql"] = str(query)
        return pd.DataFrame(columns=["ts", "symbol", "open", "close", "low", "high", "volume", "trades", "quarter"])

    monkeypatch.setattr(loader.pd, "read_sql", fake_read_sql)
    loader.load_trades(engine=FakeEngine(), symbol="BTCUSDT")

    assert "quarter IS NULL" in captured_query["sql"]

def test_funding_is_converted_from_percent_to_fraction(monkeypatch):
    fake_df = pd.DataFrame({
        "ts": [1700000000],
        "symbol": ["BTCUSDT"],
        "mark_price": [60000.0],
        "index_price": [60001.0],
        "estimated_settle_price": [0.0],
        "interest": [0.0],
        "funding": [0.01],
        "time_left_seconds": [14400],
    })

    def fake_read_sql(query, engine, params=None):
        return fake_df.copy()

    monkeypatch.setattr(loader.pd, "read_sql", fake_read_sql)
    result = loader.load_funding(engine=FakeEngine(), symbol="BTCUSDT")

    # 0.01% as a fraction is 0.0001
    assert result["funding"].iloc[0] == pytest.approx(0.0001)
