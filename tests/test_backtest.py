from types import SimpleNamespace

import numpy as np
import pandas as pd

from bot.backtest import valid_sharpes


def test_valid_sharpes_drops_inf_nan_and_thin_trade_counts():
    idx = pd.MultiIndex.from_tuples(
        [(5, 10), (5, 20), (10, 20), (10, 30)], names=["fast_window", "slow_window"]
    )
    fake_pf = SimpleNamespace(
        sharpe_ratio=lambda: pd.Series([np.inf, np.nan, 2.0, 0.8], index=idx),
        trades=SimpleNamespace(count=lambda: pd.Series([9, 0, 2, 12], index=idx)),
    )
    scores = valid_sharpes(fake_pf, min_trades=5)
    # inf dropped, NaN dropped, the 2.0 dropped for having only 2 trades
    assert list(scores.index) == [(10, 30)]
    assert scores.iloc[0] == 0.8