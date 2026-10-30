"""Walk-forward validation: re-optimize on a rolling window, trade the next one.

Every test window is traded with parameters chosen using only data that
precedes it. The test windows are stitched into one continuous out-of-sample
equity curve, so positions carry across fold boundaries like they would live.

    python -m bot.walkforward --train-years 5 --test-years 1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from bot.backtest import FREQ, INIT_CASH, compare, portfolio, sweep, valid_sharpes
from bot.data import load_ohlcv
from bot.strategy import sma_state


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive; the test window starts here
    test_end: pd.Timestamp  # exclusive


def make_folds(
    index: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = False,
) -> list[Fold]:
    """Calendar-based folds. Rolling by default; anchored keeps the start fixed.

    The test windows tile the timeline back to back, and the last one runs to
    the end of the data (it may be a partial window).
    """
    folds: list[Fold] = []
    train_start = index[0]
    train_end = train_start + pd.DateOffset(years=train_years)
    while train_end <= index[-1]:
        test_end = train_end + pd.DateOffset(years=test_years)
        folds.append(Fold(train_start, train_end, test_end))
        train_end = test_end
        train_start = (
            index[0] if anchored else train_end - pd.DateOffset(years=train_years)
        )
    return folds


def walk_forward(
    df: pd.DataFrame,
    windows: np.ndarray,
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = False,
) -> tuple[vbt.Portfolio, pd.DataFrame, pd.DataFrame]:
    """Returns (stitched OOS portfolio, per-fold table, OOS slice of df)."""
    idx, close = df.index, df["close"]
    in_pos = pd.Series(False, index=idx)
    rows: list[dict] = []

    for f in make_folds(idx, train_years, test_years, anchored):
        train = df.loc[(idx >= f.train_start) & (idx < f.train_end)]
        test_mask = (idx >= f.train_end) & (idx < f.test_end)
        if not test_mask.any():
            continue

        scores = valid_sharpes(sweep(train, windows))
        fast = slow = None
        train_sharpe = np.nan
        if not scores.empty:  # nothing trustworthy in this fold -> stay flat
            fast, slow = (int(w) for w in scores.idxmax())
            train_sharpe = float(scores.max())
            # Computed on full history so the MAs are warm; still backward-looking.
            in_pos.loc[test_mask] = sma_state(close, fast, slow).loc[test_mask]

        rows.append(
            {
                "train_start": f.train_start.date(),
                "test_start": idx[test_mask][0],
                "test_end": idx[test_mask][-1],
                "fast": fast,
                "slow": slow,
                "train_sharpe": train_sharpe,
            }
        )

    oos_start = rows[0]["test_start"]
    oos = df.loc[oos_start:]
    pos = in_pos.loc[oos_start:]
    prev = pos.shift(1, fill_value=False)
    # State changes become signals; portfolio() fills them at the next open.
    pf = portfolio(oos, pos & ~prev, ~pos & prev)

    strat_ret = pf.returns()
    # Start at the OOS span's first close, like the headline benchmark: no return
    # from the bar *before* the span (it belongs to the training window).
    bh_ret = oos["close"].pct_change().fillna(0.0)
    for r in rows:
        a, b = r["test_start"], r["test_end"]
        r["strategy_%"] = ((1 + strat_ret.loc[a:b]).prod() - 1) * 100
        r["buy_hold_%"] = ((1 + bh_ret.loc[a:b]).prod() - 1) * 100
        r["test_start"], r["test_end"] = a.date(), b.date()

    table = pd.DataFrame(rows)
    table[["fast", "slow"]] = table[["fast", "slow"]].astype("Int64")
    return pf, table, oos


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--train-years", type=int, default=5)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--anchored", action="store_true", help="expanding train window")
    p.add_argument(
        "--holdout-years",
        type=int,
        default=3,
        help="most recent N years are reserved and never touched (default 3)",
    )
    p.add_argument(
        "--use-holdout",
        action="store_true",
        help="include the holdout period. Final exam: run once, after the strategy is frozen",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    df = load_ohlcv(args.symbol)
    if args.holdout_years > 0 and not args.use_holdout:
        cutoff = df.index[-1] - pd.DateOffset(years=args.holdout_years)
        df = df.loc[:cutoff]
        print(f"Holdout reserved: data after {cutoff.date()} is not used.\n")

    pf, table, oos = walk_forward(
        df,
        np.arange(5, 205, 5),
        train_years=args.train_years,
        test_years=args.test_years,
        anchored=args.anchored,
    )

    print(table.to_string(index=False, float_format="{:.2f}".format))
    n_pairs = len(table[["fast", "slow"]].drop_duplicates())
    mode = "anchored" if args.anchored else "rolling"
    holdout = "included" if args.use_holdout else f"{args.holdout_years}y reserved"
    print(f"\nConfig: train={args.train_years}y test={args.test_years}y {mode}, holdout {holdout}")
    print(f"{len(table)} folds, {n_pairs} distinct parameter pairs chosen")
    print(f"OOS period: {oos.index[0].date()} to {oos.index[-1].date()}\n")

    bench = vbt.Portfolio.from_holding(oos["close"], init_cash=INIT_CASH, freq=FREQ)
    print(compare(pf, bench))


if __name__ == "__main__":
    main()