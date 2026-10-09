"""
Fetch historical bars into the local cache (Phase 1).

Primary source: yfinance — free, no API key. We pull SPLIT-adjusted but DIVIDEND-UNADJUSTED
prices (auto_adjust=False, then keep the plain 'close'):
  • split-adjusted   → no fake gaps at splits (NVDA 40x, AMZN 20:1, TSLA…), so moving averages
                       stay continuous, AND
  • dividend-UNadjusted → the actual price the stock traded at, which is what an option's
                       intrinsic value (spot − strike) keys off — the right basis for pricing
                       options in the backtest (a dividend-adjusted series would misprice them).
This also matches the live bot's raw Alpaca feed closely (verified: SPY EMAs/ADX reproduce the
live log). Daily has no history limit; intraday is bounded by yfinance (hourly ~730d, minute
~30d) — enforced here so the intraday path is ready to use later without surprises.

Run (pull the full universe, ~6 yrs of tradeable history + warmup):
    python -m backtest.data.fetch                       # daily, 2018-01-01 -> today
    python -m backtest.data.fetch --granularity 1h --start 2024-01-01
    python -m backtest.data.fetch --symbols AAPL MSFT --force
"""
import argparse
import time
from datetime import date, datetime, timedelta
from typing import List, Optional
import pandas as pd
import yfinance as yf

from backtest import universe
from backtest.data import store

# Default raw-history start. Tradeable window is ~2020->now; the extra ~2 years is warmup for
# the 200-day regime SMA and the 252-day vol-rank the entry filters need before the first trade.
DEFAULT_START = "2018-01-01"


def fetch_bars(symbol: str, start: str, end: Optional[str] = None,
               granularity: str = "1d") -> Optional[pd.DataFrame]:
    """Pull adjusted bars for one symbol from yfinance. Clamps intraday requests to yfinance's
    allowed lookback. Returns a normalized OHLCV frame (may be shorter than requested if the
    name IPO'd later) or None on failure/empty."""
    spec = store.GRANULARITIES[granularity]
    end = end or date.today().isoformat()
    max_days = spec["max_days"]
    if max_days is not None:
        earliest = (datetime.now() - timedelta(days=max_days - 1)).date().isoformat()
        if start < earliest:
            start = earliest  # yfinance would reject/short-return an over-long intraday window
    try:
        df = yf.Ticker(symbol).history(
            start=start, end=end, interval=spec["yf_interval"],
            auto_adjust=False, actions=False, raise_errors=False,
        )
        if df is None or df.empty:
            return None
        return store.normalize(df)
    except Exception as e:
        print(f"  ! {symbol}: fetch failed ({e})")
        return None


def pull_universe(symbols: Optional[List[str]] = None, start: str = DEFAULT_START,
                  end: Optional[str] = None, granularity: str = "1d",
                  force: bool = False, pause: float = 0.4) -> List[store.Coverage]:
    """Fetch+cache each symbol (skipping ones already cached unless force). Returns coverage."""
    symbols = symbols or universe.symbols()
    label = store.GRANULARITIES[granularity]["label"]
    print(f"Pulling {len(symbols)} symbols | {label} | {start} -> {end or 'today'} "
          f"| force={force}\n")
    ok = skip = fail = 0
    for i, sym in enumerate(symbols, 1):
        if not force and store.has_bars(sym, granularity):
            print(f"[{i:>2}/{len(symbols)}] {sym:<5} cached — skip")
            skip += 1
            continue
        df = fetch_bars(sym, start, end, granularity)
        if df is None or df.empty:
            print(f"[{i:>2}/{len(symbols)}] {sym:<5} NO DATA")
            fail += 1
            continue
        e = store.save_bars(sym, df, granularity, source="yfinance", adjustment="split")
        print(f"[{i:>2}/{len(symbols)}] {sym:<5} {e['rows']:>5} bars  {e['start']} -> {e['end']}")
        ok += 1
        time.sleep(pause)  # be polite to yfinance
    print(f"\nDone: {ok} fetched, {skip} cached-skipped, {fail} failed.")
    return store.coverage(granularity)


def _parse_args():
    ap = argparse.ArgumentParser(description="Fetch historical bars into the backtest cache.")
    ap.add_argument("--granularity", default="1d", choices=list(store.GRANULARITIES))
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", nargs="*", default=None, help="subset (default: full universe)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cov = pull_universe(a.symbols, a.start, a.end, a.granularity, a.force)
    total_rows = sum(c.rows for c in cov)
    print(f"\nCache now holds {len(cov)} symbols / {total_rows:,} {a.granularity} bars.")
