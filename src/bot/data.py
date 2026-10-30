"""Market data loading with a local parquet cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlcv(
    symbol: str,
    start: str = "2005-01-01",
    end: str | None = None,
    interval: str = "1d",
    refresh: bool = False,
    cache_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Return split/dividend-adjusted OHLCV with lowercase columns.

    The cache holds whatever was fetched first; pass refresh=True to re-download
    (e.g. to extend history or pick up new bars).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol.upper()}_{interval}.parquet"

    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        raw = yf.Ticker(symbol).history(
            start=start, end=end, interval=interval, auto_adjust=True
        )
        if raw.empty:
            raise ValueError(f"No data returned for {symbol!r}")
        df = raw.rename(columns=str.lower)[COLUMNS].astype("float64")
        idx = pd.DatetimeIndex(raw.index).tz_localize(None)
        df.index = idx.normalize() if interval == "1d" else idx
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(path)

    return df.loc[start:end]