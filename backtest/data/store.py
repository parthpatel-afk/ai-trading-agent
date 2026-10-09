"""
Local cache of historical bars — parquet, tagged by granularity.

Layout (all under the project root, git-ignored):
    backtest/data/cache/
        1d/AAPL.parquet   1d/MSFT.parquet   ...     (daily — what we use now)
        1h/AAPL.parquet   ...                         (hourly — future intraday strategies)
        1m/...                                         (minute — future)
        manifest.json                                 (per-symbol coverage + provenance)

Everything downstream loads bars through `load_bars(symbol, granularity)`, so adding a new
granularity later needs zero changes elsewhere — you just fetch into a new sub-folder.
Bars are normalized to lowercase OHLCV columns, a tz-naive DatetimeIndex, oldest→newest —
matching what the live pipeline/indicators expect.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd

from agent.paths import project_root

# Supported granularities → (yfinance interval, human label, max lookback days yfinance allows).
# None = no practical limit (daily). Intraday limits are yfinance's, enforced by the fetcher.
GRANULARITIES: Dict[str, dict] = {
    "1d": {"yf_interval": "1d", "label": "daily",  "max_days": None},
    "1h": {"yf_interval": "1h", "label": "hourly", "max_days": 730},
    "1m": {"yf_interval": "1m", "label": "minute", "max_days": 30},
}

OHLCV = ["open", "high", "low", "close", "volume"]


def cache_root():
    return project_root() / "backtest" / "data" / "cache"


def bars_dir(granularity: str = "1d"):
    d = cache_root() / granularity
    d.mkdir(parents=True, exist_ok=True)
    return d


def bars_path(symbol: str, granularity: str = "1d"):
    return bars_dir(granularity) / f"{symbol.upper()}.parquet"


def has_bars(symbol: str, granularity: str = "1d") -> bool:
    return bars_path(symbol, granularity).exists()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase OHLCV columns, tz-naive DatetimeIndex, sorted oldest→newest, no NaN rows."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in OHLCV if c in df.columns]
    df = df[keep]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(how="any")


def save_bars(symbol: str, df: pd.DataFrame, granularity: str = "1d",
              source: str = "yfinance", adjustment: str = "split") -> dict:
    """Persist bars + update the manifest. Returns this symbol's manifest entry.
    `adjustment`: 'split' = split-adjusted but dividend-UNadjusted (the actual traded price,
    correct for option intrinsic); 'split+dividend' = total-return series."""
    df = normalize(df)
    df.to_parquet(bars_path(symbol, granularity))
    entry = {
        "symbol": symbol.upper(), "granularity": granularity, "source": source,
        "adjustment": adjustment, "rows": int(len(df)),
        "start": str(df.index[0].date()) if len(df) else None,
        "end": str(df.index[-1].date()) if len(df) else None,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_manifest_entry(entry)
    return entry


def load_bars(symbol: str, granularity: str = "1d") -> Optional[pd.DataFrame]:
    """Load cached bars (normalized) or None if not cached."""
    p = bars_path(symbol, granularity)
    if not p.exists():
        return None
    return normalize(pd.read_parquet(p))


# ── Manifest (coverage + provenance) ─────────────────────────────────

def manifest_path():
    return cache_root() / "manifest.json"


def read_manifest() -> Dict[str, dict]:
    p = manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _write_manifest_entry(entry: dict) -> None:
    m = read_manifest()
    m[f"{entry['granularity']}/{entry['symbol']}"] = entry
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(m, indent=2, sort_keys=True))


@dataclass
class Coverage:
    symbol: str
    granularity: str
    rows: int
    start: Optional[str]
    end: Optional[str]


def coverage(granularity: str = "1d") -> List[Coverage]:
    """Summarize what's cached for a granularity (from the manifest)."""
    out = []
    for key, e in sorted(read_manifest().items()):
        if e.get("granularity") == granularity:
            out.append(Coverage(e["symbol"], granularity, e["rows"], e.get("start"), e.get("end")))
    return out
