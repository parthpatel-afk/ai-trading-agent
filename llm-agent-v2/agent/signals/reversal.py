"""
AI Investor Agent - Reversal / Exhaustion Signal Module (Phase 4)

Detects when a trend is getting *tired* and a turn is likely, in BOTH directions
(MASTER_STRATEGY.md §3.1/§3.3, REVERSAL_STRATEGY_PLAN.md). Used to TIME the directional
satellite buys (buy the bounce at an exhausted bottom / the rejection at an exhausted top)
instead of chasing a move that's already happened.

Three classic, well-documented tells, combined with a confirmation requirement:
  1. Momentum divergence (RSI): price makes a new extreme but RSI does not → energy fading.
  2. Volume climax + fade (Wyckoff-lite): a high-volume spike then volume drying up.
  3. Fibonacci retracement levels (38.2 / 50 / 61.8%) for context on where turns cluster.

Discipline (per the plan): never act on a single raw flag — require a divergence AND the
turn actually starting (a 1-bar reversal in the right direction). The caller additionally
gates by IV environment + macro regime.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import ta
from loguru import logger


@dataclass
class ReversalSignals:
    symbol: str
    bearish_divergence: bool = False   # price higher-high, RSI lower-high → top forming
    bullish_divergence: bool = False   # price lower-low, RSI higher-low → bottom forming
    volume_climax_fade: bool = False   # a recent volume spike that is now fading
    near_fib_support: bool = False     # price near a 38.2/50/61.8 retracement from below
    near_fib_resistance: bool = False  # price near a retracement level from above
    fib_levels: Dict[str, float] = field(default_factory=dict)
    reversal_up: bool = False          # confirmed bullish reversal setup (→ buy calls)
    reversal_down: bool = False        # confirmed bearish reversal setup (→ buy puts)
    notes: str = ""


def _find_pivots(values: np.ndarray, left: int = 3, right: int = 3, kind: str = "high") -> List[int]:
    """Indices of local pivots: a bar higher (or lower) than `left` bars before and `right`
    bars after it. Simple, dependency-free swing detector."""
    pivots = []
    n = len(values)
    for i in range(left, n - right):
        window = values[i - left:i + right + 1]
        if kind == "high" and values[i] == window.max() and np.argmax(window) == left:
            pivots.append(i)
        elif kind == "low" and values[i] == window.min() and np.argmin(window) == left:
            pivots.append(i)
    return pivots


def compute_reversal_signals(symbol: str, df: pd.DataFrame, lookback: int = 60) -> ReversalSignals:
    """Compute reversal/exhaustion signals from a daily OHLCV DataFrame."""
    sig = ReversalSignals(symbol=symbol)
    try:
        if df is None or len(df) < 40:
            return sig
        df = df.copy().tail(lookback)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

        # ── 1. Divergence (use the two most recent confirmed pivots) ──
        price_highs = _find_pivots(high.values, kind="high")
        price_lows = _find_pivots(low.values, kind="low")
        rsi_vals = rsi.values

        if len(price_highs) >= 2:
            a, b = price_highs[-2], price_highs[-1]
            if high.values[b] > high.values[a] and rsi_vals[b] < rsi_vals[a]:
                sig.bearish_divergence = True
                sig.notes += "bearish RSI divergence (price HH, RSI LH). "
        if len(price_lows) >= 2:
            a, b = price_lows[-2], price_lows[-1]
            if low.values[b] < low.values[a] and rsi_vals[b] > rsi_vals[a]:
                sig.bullish_divergence = True
                sig.notes += "bullish RSI divergence (price LL, RSI HL). "

        # ── 2. Volume climax + fade ───────────────────────────────────
        vol_avg = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
        recent_vol = volume.tail(8)
        if vol_avg > 0 and float(recent_vol.max()) >= 2.0 * vol_avg:
            peak_pos = int(np.argmax(recent_vol.values))
            # fade = volume after the spike is lower than at the spike
            if peak_pos < len(recent_vol) - 1 and float(recent_vol.iloc[-1]) < float(recent_vol.iloc[peak_pos]):
                sig.volume_climax_fade = True
                sig.notes += "volume climax then fade. "

        # ── 3. Fibonacci retracement of the most recent swing ─────────
        swing_high = float(high.tail(lookback).max())
        swing_low = float(low.tail(lookback).min())
        rng = swing_high - swing_low
        last = float(close.iloc[-1])
        if rng > 0:
            levels = {
                "0.382": round(swing_high - 0.382 * rng, 2),
                "0.5": round(swing_high - 0.5 * rng, 2),
                "0.618": round(swing_high - 0.618 * rng, 2),
            }
            sig.fib_levels = levels
            tol = 0.02 * last  # within 2% of a level
            for lv in levels.values():
                if abs(last - lv) <= tol:
                    # near a level from below = resistance; from above = support
                    if last <= lv:
                        sig.near_fib_resistance = True
                    else:
                        sig.near_fib_support = True

        # ── Confirmation: divergence AND the turn actually starting ───
        # (don't act on the divergence alone — it can persist for weeks).
        turning_up = float(close.iloc[-1]) > float(close.iloc[-2])
        turning_down = float(close.iloc[-1]) < float(close.iloc[-2])
        sig.reversal_up = bool(sig.bullish_divergence and turning_up)
        sig.reversal_down = bool(sig.bearish_divergence and turning_down)

        if sig.reversal_up or sig.reversal_down:
            logger.debug(f"{symbol}: reversal {'UP' if sig.reversal_up else 'DOWN'} — {sig.notes.strip()}")
        return sig
    except Exception as e:
        logger.debug(f"{symbol}: reversal computation failed: {e}")
        return sig
