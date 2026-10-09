"""
Trend-Buyer v1 — Indicator math (pure functions on a daily OHLCV DataFrame).

Hand-rolled with pandas/numpy (no `ta`/scipy dependency) so the bot stays light and
the formulas are auditable. ATR/ADX use Wilder's smoothing (RMA), the canonical method.
All functions expect lowercase columns: open, high, low, close, volume — oldest→newest.
"""
from typing import Optional
import numpy as np
import pandas as pd


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA) — ewm with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Latest Average True Range (in price units). None if insufficient data."""
    if df is None or len(df) < period + 1:
        return None
    a = _wilder_rma(true_range(df), period)
    val = a.iloc[-1]
    return float(val) if pd.notna(val) else None


def adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Latest ADX (0-100 trend-strength meter). None if insufficient data."""
    if df is None or len(df) < 2 * period + 1:
        return None
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)

    atr_s = _wilder_rma(tr, period)
    plus_di = 100 * _wilder_rma(pd.Series(plus_dm, index=df.index), period) / atr_s
    minus_di = 100 * _wilder_rma(pd.Series(minus_dm, index=df.index), period) / atr_s
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx_s = _wilder_rma(dx.fillna(0), period)
    val = adx_s.iloc[-1]
    return float(val) if pd.notna(val) else None


def vol_rank(close: pd.Series, hv_window: int = 20, lookback: int = 252) -> Optional[float]:
    """
    Cheap/expensive vol gauge in [0,100]. We compute the underlying's historical
    volatility (HV) as a proxy for implied vol, then rank today's HV within its own
    ~1-year range:  (current - min) / (max - min) * 100.   0 = cheapest, 100 = priciest.
    """
    if close is None or len(close) < hv_window + 5:
        return None
    log_ret = np.log(close / close.shift(1))
    hv = log_ret.rolling(hv_window).std() * np.sqrt(252)
    hv = hv.dropna()
    if hv.empty:
        return None
    window = hv.iloc[-lookback:] if len(hv) > lookback else hv
    lo, hi, cur = float(window.min()), float(window.max()), float(hv.iloc[-1])
    if hi <= lo:
        return 50.0
    return round(100.0 * (cur - lo) / (hi - lo), 1)
