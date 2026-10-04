import pandas as pd

from control.baselines import BuyAndHold
from control.controller_base import Controller


def test_buy_and_hold_is_constant():
    bh = BuyAndHold(position=0.7)
    assert bh.act(pd.Series({"log_return": 0.1})) == 0.7
    assert bh.act(pd.Series({"log_return": -0.5})) == 0.7


def test_buy_and_hold_satisfies_controller_protocol():
    assert isinstance(BuyAndHold(), Controller)
