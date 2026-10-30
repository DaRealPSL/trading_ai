import pandas as pd
import pytest

from bot.strategy import sma_crossover


def test_one_entry_one_exit():
    close = pd.Series([1, 1, 1, 1, 5, 5, 5, 1, 1, 1], dtype=float)
    entries, exits = sma_crossover(close, fast=2, slow=3)
    assert entries.sum() == 1 and entries.idxmax() == 4
    assert exits.sum() == 1 and exits.idxmax() == 7


def test_no_signals_during_warmup():
    close = pd.Series(range(1, 30), dtype=float)
    entries, exits = sma_crossover(close, fast=3, slow=10)
    assert not entries.iloc[:10].any() and not exits.iloc[:10].any()


def test_rejects_bad_windows():
    with pytest.raises(ValueError):
        sma_crossover(pd.Series([1.0, 2.0]), fast=5, slow=3)


def test_state_is_long_while_fast_is_strictly_above_slow():
    from bot.strategy import sma_state

    close = pd.Series([1, 1, 1, 1, 5, 5, 5, 1, 1, 1], dtype=float)
    state = sma_state(close, fast=2, slow=3)
    assert not state.iloc[:4].any()  # warm-up / flat before the up-move
    assert state.iloc[4:6].all()  # long from the entry bar (4)
    # bar 6: both SMAs equal 5, so "strictly above" is False. The crossover
    # exit event fires one bar later (7); the state flips on the tie bar.
    assert not state.iloc[6:].any()