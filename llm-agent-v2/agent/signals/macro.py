"""
AI Investor Agent - Macro / Market Regime Signal Module
Detects the current market environment to adjust agent aggression.
Uses VIX from yfinance (free) and broad market trend via SPY.
"""
from dataclasses import dataclass
from typing import Optional
import yfinance as yf
import pandas as pd
from loguru import logger

from agent.config import config


@dataclass
class MacroSignals:
    """Market regime and macro environment signals."""
    vix_level: Optional[float] = None
    vix_change_1d: Optional[float] = None       # 1-day VIX change %
    regime: str = "unknown"                      # "calm_bull", "trending_bull", "volatile", "bear", "crisis"
    regime_score: float = 0.5                   # 0=worst, 1=best for trading
    spy_trend: str = "neutral"                  # "uptrend", "downtrend", "neutral"
    spy_above_50ema: bool = False
    spy_above_200ema: bool = False
    market_breadth: str = "unknown"
    position_size_multiplier: float = 1.0       # Reduce sizes in bad regimes
    allow_new_entries: bool = True              # False in crisis
    notes: str = ""


def compute_macro_signals() -> MacroSignals:
    """
    Compute macro regime using free data sources:
    - VIX from yfinance (^VIX)
    - SPY trend for market direction
    """
    signals = MacroSignals()

    try:
        # ---- VIX ----
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d", interval="1d")
        if vix_hist is not None and len(vix_hist) >= 2:
            current_vix = float(vix_hist["Close"].iloc[-1])
            prev_vix = float(vix_hist["Close"].iloc[-2])
            signals.vix_level = round(current_vix, 2)
            signals.vix_change_1d = round((current_vix - prev_vix) / prev_vix * 100, 2)
        else:
            current_vix = 20.0  # Assume neutral if unavailable
            signals.vix_level = current_vix
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
        current_vix = 20.0

    try:
        # ---- SPY Trend ----
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="60d", interval="1d")
        if spy_hist is not None and len(spy_hist) >= 50:
            close = spy_hist["Close"]
            ema_50 = close.ewm(span=50).mean().iloc[-1]
            ema_200_available = len(close) >= 200
            if ema_200_available:
                ema_200 = close.ewm(span=200).mean().iloc[-1]
                signals.spy_above_200ema = bool(close.iloc[-1] > ema_200)

            current_spy = float(close.iloc[-1])
            signals.spy_above_50ema = bool(current_spy > ema_50)

            # 10-day trend
            spy_10d_return = (current_spy - float(close.iloc[-10])) / float(close.iloc[-10])
            if spy_10d_return > 0.02:
                signals.spy_trend = "uptrend"
            elif spy_10d_return < -0.02:
                signals.spy_trend = "downtrend"
            else:
                signals.spy_trend = "neutral"
    except Exception as e:
        logger.warning(f"SPY trend fetch failed: {e}")

    # ---- Regime Classification ----
    # Based on VIX level + SPY trend
    cfg_risk = config.risk

    if current_vix >= cfg_risk.vix_extreme_threshold:
        signals.regime = "crisis"
        signals.regime_score = 0.0
        signals.position_size_multiplier = 0.25
        signals.allow_new_entries = False
        signals.notes = f"VIX={current_vix:.1f} — CRISIS mode. New entries paused. Protecting capital."

    elif current_vix >= cfg_risk.vix_high_threshold:
        signals.regime = "volatile"
        signals.regime_score = 0.35
        signals.position_size_multiplier = 0.50
        signals.allow_new_entries = True
        signals.notes = f"VIX={current_vix:.1f} — Elevated volatility. Reduced position sizes (50%)."

    elif current_vix >= 18:
        if signals.spy_trend == "uptrend":
            signals.regime = "trending_bull"
            signals.regime_score = 0.75
            signals.position_size_multiplier = 0.90
            signals.notes = f"VIX={current_vix:.1f} — Moderate vol, bullish trend. Slightly cautious."
        else:
            signals.regime = "choppy"
            signals.regime_score = 0.50
            signals.position_size_multiplier = 0.70
            signals.notes = f"VIX={current_vix:.1f} — Choppy market. Selective entries only."

    else:  # VIX < 18 = calm
        if signals.spy_trend == "uptrend" and signals.spy_above_50ema:
            signals.regime = "calm_bull"
            signals.regime_score = 1.0
            signals.position_size_multiplier = 1.0
            signals.notes = f"VIX={current_vix:.1f} — Ideal conditions. Full position sizes."
        elif signals.spy_trend == "downtrend":
            signals.regime = "bear"
            signals.regime_score = 0.25
            signals.position_size_multiplier = 0.40
            signals.allow_new_entries = True
            signals.notes = f"VIX={current_vix:.1f} — Bear trend despite low vol. Heavily reduced sizing."
        else:
            signals.regime = "calm_neutral"
            signals.regime_score = 0.65
            signals.position_size_multiplier = 0.85
            signals.notes = f"VIX={current_vix:.1f} — Calm market, no clear trend."

    logger.info(f"Macro: VIX={signals.vix_level}, Regime={signals.regime}, SizeMult={signals.position_size_multiplier}")
    return signals
