"""Read-only audit of the walk-forward pipeline. Changes no strategy logic.

    python -m bot.audit --symbol QQQ --train-years 3

Sections: (A) why Sharpe is non-finite, (B) which folds each CLI config builds,
(C) end-to-end look-ahead test, (D) benchmark vs strategy conventions.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import vectorbt as vbt

from bot.backtest import FREQ, INIT_CASH, SLIPPAGE, portfolio, sweep
from bot.data import load_ohlcv
from bot.walkforward import build_parser, make_folds, walk_forward

WINDOWS = np.arange(5, 205, 5)


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def classify(r: pd.Series, n_trades: int) -> str:
    """Why can't this return series produce a finite Sharpe?"""
    if n_trades == 0:
        return "no trades"
    nonzero = int((r != 0).sum())
    if nonzero == 0:
        return "all-zero returns"
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return f"zero variance ({nonzero} nonzero bars)"
    if nonzero <= 3:
        return f"only {nonzero} nonzero return bars"
    return "other (std > 0, check numerics)"


def audit_nonfinite_sharpe(df, a, show: int = 10) -> None:
    header("A. Non-finite Sharpe in training windows")
    idx = df.index
    rows = []
    tot = dict(combos=0, nan=0, nan_with_trades=0, pos_inf=0, neg_inf=0)
    for k, f in enumerate(make_folds(idx, a.train_years, a.test_years, a.anchored), 1):
        if not ((idx >= f.train_end) & (idx < f.test_end)).any():
            continue
        train = df.loc[(idx >= f.train_start) & (idx < f.train_end)]
        pf = sweep(train, WINDOWS)
        sharpe, trades = pf.sharpe_ratio(), pf.trades.count()
        in_mkt, rets, sortino = (pf.assets() > 0).sum(), pf.returns(), pf.sortino_ratio()
        tot["combos"] += len(sharpe)
        tot["nan"] += int(sharpe.isna().sum())
        tot["nan_with_trades"] += int((sharpe.isna() & (trades > 0)).sum())
        tot["pos_inf"] += int((sharpe == np.inf).sum())
        tot["neg_inf"] += int((sharpe == -np.inf).sum())
        for col in sharpe.index[np.isinf(sharpe)]:
            r = rets[col]
            rows.append(
                {
                    "fold": k,
                    "fast": col[0],
                    "slow": col[1],
                    "sharpe": sharpe[col],
                    "trades": int(trades[col]),
                    "bars_in_mkt": int(in_mkt[col]),
                    "nonzero_bars": int((r != 0).sum()),
                    "ret_mean": float(r.mean()),
                    "ret_std": float(r.std(ddof=1)),
                    "sortino": sortino[col],
                    "cause": classify(r, int(trades[col])),
                }
            )
    print("totals:", tot)
    if not rows:
        print("No infinite Sharpe in any training window for this config.")
        return
    out = pd.DataFrame(rows).sort_values("trades")
    print(f"\n{len(out)} infinite-Sharpe combos. Representative rows (fewest trades first):")
    print(out.head(show).to_string(index=False, float_format="{:.3g}".format))
    print("\ncauses:\n" + out["cause"].value_counts().to_string())
    print("\ntrade counts of the infinite combos:\n" + out["trades"].describe().to_string())


def audit_configs(df, holdout_years: int) -> None:
    header("B. What each CLI configuration actually builds")
    idx = df.index
    configs = ["", "--train-years 3", "--train-years 8", "--test-years 2", "--anchored"]
    rows = []
    for c in configs:
        a = build_parser().parse_args(c.split())
        folds = make_folds(idx, a.train_years, a.test_years, a.anchored)
        rows.append(
            {
                "args": c or "(defaults)",
                "train_y": a.train_years,
                "test_y": a.test_years,
                "anchored": a.anchored,
                "folds": len(folds),
                "first_train": folds[0].train_start.date(),
                "first_test": folds[0].train_end.date(),
                "last_test_end": min(folds[-1].test_end, idx[-1]).date(),
                "starts_vary": len({f.train_start for f in folds}) > 1,
            }
        )
    print(f"data used: {idx[0].date()} to {idx[-1].date()} (holdout {holdout_years}y excluded)")
    print(pd.DataFrame(rows).to_string(index=False))
    # structural invariants for every config
    for c in configs:
        a = build_parser().parse_args(c.split())
        folds = make_folds(idx, a.train_years, a.test_years, a.anchored)
        assert all(f.train_start < f.train_end < f.test_end for f in folds)
        assert all(x.test_end == y.train_end for x, y in zip(folds, folds[1:]))
    print("invariants OK: train < test, test windows contiguous and non-overlapping")


def audit_leakage(df, a, frac: float = 0.6) -> None:
    header("C. Look-ahead test: scramble the future, the past must not move")
    cutoff = df.index[int(len(df) * frac)]
    base_pf, base_tbl, _ = walk_forward(df, WINDOWS, a.train_years, a.test_years, a.anchored)

    rng = np.random.default_rng(0)
    shock = pd.Series(1.0, index=df.index)
    after = df.index > cutoff
    shock[after] = np.exp(rng.normal(0, 0.05, int(after.sum()))).cumprod()
    df2 = df.copy()
    for c in ["open", "high", "low", "close"]:
        df2[c] = df[c] * shock
    pf2, tbl2, _ = walk_forward(df2, WINDOWS, a.train_years, a.test_years, a.anchored)

    v1, v2 = base_pf.value().loc[:cutoff], pf2.value().loc[:cutoff]
    same_value = bool(np.allclose(v1.to_numpy(), v2.to_numpy()))
    early = base_tbl["test_start"] <= cutoff.date()  # folds whose training ended <= cutoff
    same_params = base_tbl.loc[early, ["fast", "slow"]].equals(tbl2.loc[early, ["fast", "slow"]])
    later = ~early
    moved = int((base_tbl.loc[later, ["fast", "slow"]].to_numpy()
                 != tbl2.loc[later, ["fast", "slow"]].to_numpy()).any(axis=1).sum())
    print(f"future scrambled after {cutoff.date()}")
    print(f"portfolio value up to cutoff identical:      {same_value}")
    print(f"params of folds trained before cutoff equal: {same_params}")
    print(f"later folds whose params did change:         {moved} of {int(later.sum())} (shows the test can detect a change)")
    print("PASS" if same_value and same_params else "FAIL: future data is influencing the past")


def audit_benchmark(df, a) -> None:
    header("D. Benchmark vs strategy conventions")
    pf, table, oos = walk_forward(df, WINDOWS, a.train_years, a.test_years, a.anchored)
    bench = vbt.Portfolio.from_holding(oos["close"], init_cash=INIT_CASH, freq=FREQ)
    first = pd.Series(False, index=oos.index)
    first.iloc[0] = True
    like = portfolio(oos, first, pd.Series(False, index=oos.index))  # strategy conventions

    print(f"same dates:  {pf.value().index.equals(bench.value().index)}  "
          f"({oos.index[0].date()} to {oos.index[-1].date()}, {len(oos)} bars)")
    print(f"strategy: fills at next open, slippage {SLIPPAGE:.2%}/side, fees 0, cash {INIT_CASH}, 0% on idle cash")
    print("benchmark (from_holding): buys at first CLOSE, no slippage, same cash")
    print(f"\nstrategy total return:                 {pf.total_return():.2%}")
    print(f"B&H as reported (from_holding):         {bench.total_return():.2%}")
    print(f"B&H with the strategy's own fill rules: {like.total_return():.2%}")
    print(f"gap between the two B&H numbers:        {(bench.total_return() - like.total_return()) * 100:.2f} pp")
    compounded = float(np.prod(1 + table["buy_hold_%"] / 100) - 1)
    print(f"per-fold B&H compounded vs headline:    {compounded:.2%} vs {bench.total_return():.2%}")


def main() -> None:
    p = build_parser()
    a = p.parse_args()
    df = load_ohlcv(a.symbol)
    if a.holdout_years > 0 and not a.use_holdout:
        df = df.loc[: df.index[-1] - pd.DateOffset(years=a.holdout_years)]
    print(f"{a.symbol}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
    audit_configs(df, a.holdout_years)
    audit_nonfinite_sharpe(df, a)
    audit_leakage(df, a)
    audit_benchmark(df, a)


if __name__ == "__main__":
    main()