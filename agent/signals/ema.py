"""
AI Investor v2 — EMA Crossover Signal Engine
The entire signal logic for the bot: 9/21 EMA crossover on the daily chart.

A BULL signal = EMA-9 crosses ABOVE EMA-21.
A BEAR signal = EMA-9 crosses BELOW EMA-21.

EMAs use the standard recursive formula (pandas ewm with adjust=False), which
matches TradingView's `ta.ema()` once warmed up — so the bot's crosses line up
with what the client sees on their chart.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
from loguru import logger

from agent.config import config


@dataclass
class EmaState:
    """EMA snapshot + most-recent crossover for one symbol (daily timeframe)."""
    symbol: str
    close: float = 0.0
    ema_fast: float = 0.0           # EMA-9
    ema_slow: float = 0.0           # EMA-21
    fast_above_slow: bool = False   # current regime: is 9 above 21 right now?
    direction: str = "neutral"      # "bull" | "bear" — desired side from current alignment

    cross: Optional[str] = None     # most recent crossover: "BULL" | "BEAR" | None
    cross_age_days: Optional[int] = None   # completed daily bars since that cross (0 = latest bar)
    fresh: bool = False             # True if the cross is within signal_lookback_days
    bars: int = 0                   # number of completed bars used


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard EMA (recursive, adjust=False) — matches TradingView ta.ema()."""
    return series.ewm(span=span, adjust=False).mean()


def drop_unclosed_daily_bar(df: Optional[pd.DataFrame], now_et: datetime) -> Optional[pd.DataFrame]:
    """
    Drop the last row if it is today's still-forming daily candle.

    The strategy acts only on CLOSED candles. When the bot runs at the open, the
    data source may already include a partial bar dated today — we remove it so the
    most recent row is always the prior session's confirmed close.
    """
    if df is None or df.empty:
        return df
    try:
        last_date = pd.Timestamp(df.index[-1]).date()
        if last_date >= now_et.date():
            return df.iloc[:-1]
    except Exception:
        pass
    return df


def compute_ema_state(symbol: str, df: pd.DataFrame) -> Optional[EmaState]:
    """
    Compute EMA-9/EMA-21 and locate the most recent crossover.

    `df` must contain only CLOSED daily bars (oldest→newest), with a 'close' column.
    Returns None if there isn't enough history.
    """
    cfg = config.strategy
    min_bars = cfg.ema_slow + 2
    if df is None or len(df) < min_bars:
        logger.warning(f"{symbol}: insufficient bars for EMA ({0 if df is None else len(df)} < {min_bars})")
        return None

    try:
        close = df["close"].astype(float)
        ema_fast = _ema(close, cfg.ema_fast)
        ema_slow = _ema(close, cfg.ema_slow)
        diff = ema_fast - ema_slow
        n = len(diff)

        fast_above = bool(diff.iloc[-1] > 0)

        # Walk backwards to find the most recent sign change in (fast - slow).
        cross: Optional[str] = None
        cross_age: Optional[int] = None
        for i in range(n - 1, 0, -1):
            prev = diff.iloc[i - 1]
            cur = diff.iloc[i]
            if prev <= 0 < cur:
                cross, cross_age = "BULL", (n - 1) - i
                break
            if prev >= 0 > cur:
                cross, cross_age = "BEAR", (n - 1) - i
                break

        fresh = cross is not None and cross_age is not None and cross_age <= cfg.signal_lookback_days

        return EmaState(
            symbol=symbol,
            close=round(float(close.iloc[-1]), 2),
            ema_fast=round(float(ema_fast.iloc[-1]), 4),
            ema_slow=round(float(ema_slow.iloc[-1]), 4),
            fast_above_slow=fast_above,
            direction="bull" if fast_above else "bear",
            cross=cross,
            cross_age_days=cross_age,
            fresh=fresh,
            bars=n,
        )
    except Exception as e:
        logger.error(f"{symbol}: EMA computation failed: {e}")
        return None
