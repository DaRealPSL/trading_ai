"""vectorbt backtests: parameter sweep in-sample, validate out-of-sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from bot.data import load_ohlcv
from bot.strategy import sma_crossover

INIT_CASH = 10_000
SLIPPAGE = 0.0005  # 5 bps per fill; SPY is commission-free at most brokers
FREQ = "1D"
MIN_TRADES = 5  # ignore parameter combos with fewer trades than this when ranking

# vectorbt annualizes by calendar time (365 days/yr by default), but stock data
# has ~252 bars/yr. Without this, Sharpe, Sortino, Calmar and annualized return
# are all inflated. Rankings don't change, absolute numbers do.
vbt.settings.returns["year_freq"] = "252 days"


def portfolio(df: pd.DataFrame, entries, exits) -> vbt.Portfolio:
    """Signals fire on bar t's close, so fill at bar t+1's open."""
    return vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries.shift(1, fill_value=False),
        exits=exits.shift(1, fill_value=False),
        price=df["open"],
        init_cash=INIT_CASH,
        slippage=SLIPPAGE,
        freq=FREQ,
    )


def sweep(df: pd.DataFrame, windows: np.ndarray) -> vbt.Portfolio:
    """Every (fast, slow) SMA pair from `windows`, in one vectorized run."""
    fast_ma, slow_ma = vbt.MA.run_combs(
        df["close"], windows, r=2, short_names=["fast", "slow"]
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return portfolio(df, entries, exits)


def valid_sharpes(pf: vbt.Portfolio, min_trades: int = MIN_TRADES) -> pd.Series:
    """Sharpe per parameter combo, minus scores that can't be trusted.

    Drops non-finite scores (inf/NaN, e.g. no variance in returns) and combos
    with fewer than `min_trades` trades. Never rank on the raw sharpe_ratio():
    idxmax() happily picks an inf.
    """
    sharpe = pf.sharpe_ratio()
    trades = pf.trades.count()
    n_inf = int(np.isinf(sharpe).sum())
    if n_inf:
        print(f"  note: ignored {n_inf} parameter combos with infinite Sharpe")
    return sharpe[np.isfinite(sharpe) & (trades >= min_trades)]


COMPARE_KEYS = [
    "Total Return [%]",
    "Max Drawdown [%]",
    "Max Drawdown Duration",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
]


def time_in_market(pf: vbt.Portfolio) -> float:
    """Percent of bars with an open position (closing-bar holdings)."""
    return float((pf.assets() > 0).mean() * 100)


def compare(strategy: vbt.Portfolio, benchmark: vbt.Portfolio) -> pd.DataFrame:
    """Key stats side by side, plus how much of the time each is invested."""
    table = pd.DataFrame(
        {
            "Strategy": strategy.stats()[COMPARE_KEYS],
            "Buy & hold": benchmark.stats()[COMPARE_KEYS],
        }
    )
    table.loc["Time in market [%]"] = [
        time_in_market(strategy),
        time_in_market(benchmark),
    ]
    return table


def main(symbol: str = "SPY", split: str = "2019-01-01") -> None:
    df = load_ohlcv(symbol)
    train = df.loc[: pd.Timestamp(split) - pd.Timedelta(days=1)]
    test = df.loc[split:]

    pf = sweep(train, np.arange(5, 205, 5))
    scores = valid_sharpes(pf)
    print(f"In-sample top 5 by Sharpe ({train.index[0].date()} to {train.index[-1].date()}):")
    print(scores.sort_values(ascending=False).head(5), "\n")

    fast, slow = (int(w) for w in scores.idxmax())
    print(f"Best in-sample pair: fast={fast}, slow={slow}\n")

    # Compute signals on full history (so the MAs are warmed up), then slice.
    entries, exits = sma_crossover(df["close"], fast, slow)
    pf_test = portfolio(test, entries.loc[split:], exits.loc[split:])
    print(f"Out-of-sample ({test.index[0].date()} to {test.index[-1].date()}):")
    bench = vbt.Portfolio.from_holding(
        test["close"], init_cash=INIT_CASH, freq=FREQ
    )
    print(compare(pf_test, bench))


if __name__ == "__main__":
    main()