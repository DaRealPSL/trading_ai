"""Pure strategy logic. No I/O, no vectorbt: shared by backtest and live."""
from __future__ import annotations

import pandas as pd


def sma_crossover(
    close: pd.Series, fast: int, slow: int
) -> tuple[pd.Series, pd.Series]:
    """Boolean (entries, exits), evaluated on each bar's close.

    The signal on bar t is only knowable after t closes, so execute on t+1.
    """
    if not 0 < fast < slow:
        raise ValueError("need 0 < fast < slow")

    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    prev_fast, prev_slow = fast_ma.shift(1), slow_ma.shift(1)

    # Comparisons against NaN are False, so warm-up bars never fire.
    entries = (fast_ma > slow_ma) & (prev_fast <= prev_slow)
    exits = (fast_ma < slow_ma) & (prev_fast >= prev_slow)
    return entries, exits


def sma_state(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """True while the fast SMA is above the slow SMA (i.e. "should be long").

    Unlike sma_crossover, this reports the *state* on every bar, which is what
    walk-forward needs: when the parameters change at a fold boundary, the
    new parameters' view of the market applies immediately.
    """
    if not 0 < fast < slow:
        raise ValueError("need 0 < fast < slow")
    return close.rolling(fast).mean() > close.rolling(slow).mean()