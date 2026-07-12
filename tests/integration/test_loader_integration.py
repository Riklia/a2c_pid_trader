import os
import pytest
from data import loader

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def engine():
    dsn = os.environ.get("DEV_DB_DSN")
    if not dsn:
        pytest.skip("DEV_DB_DSN environment variable is not set. SKIPPING INTEGRATION TEST")
    return loader.get_engine(loader.DBConfig(dsn=dsn))

def test_load_trades_returns_nonempty_for_btcusdt(engine):
    df = loader.load_trades(engine, symbol="BTCUSDT",
                             start_ts=1700000000, end_ts=1700003600)
    assert not df.empty
    assert (df["open"] > 0).all()