"""
AI Investor Agent - Technical Signals Module
Computes all technical indicators from OHLCV data.
Outputs a clean, normalized signal object.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import ta
from loguru import logger

from agent.config import config


@dataclass
class TechnicalSignals:
    """
    All technical indicator values + derived signals for one symbol.
    Fully typed for reliability and dashboard display.
    """
    symbol: str

    # Price
    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    volume_avg_20: float = 0.0
    volume_ratio: float = 0.0      # Current / 20d avg

    # Trend
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    above_ema_9_21: bool = False    # Price > EMA 9 and 21
    above_ema_50: bool = False      # Price > EMA 50 (bullish trend)
    ema_9_above_21: bool = False    # Short EMA above long (trend aligned)

    # Momentum
    rsi: Optional[float] = None
    rsi_signal: str = "neutral"     # "oversold", "neutral", "overbought"

    # MACD
    macd: Optional[float] = None
    macd_signal_line: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_crossover: bool = False    # Bullish crossover in last 7 bars
    macd_bullish: bool = False      # MACD above signal (positive momentum)
    macd_cross_down: bool = False   # Bearish crossover (MACD crossed below signal) in last 7 bars
    macd_bearish: bool = False      # MACD below signal (negative momentum)

    # Volatility
    atr: Optional[float] = None
    atr_pct: Optional[float] = None  # ATR as % of close price
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_mid: Optional[float] = None
    bollinger_pct: Optional[float] = None  # 0=lower band, 1=upper band
    near_lower_band: bool = False

    # Stochastic
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    stoch_bullish: bool = False

    # Composite score (0-1)
    technical_score: float = 0.0
    entry_signal: bool = False
    exit_signal: bool = False
    exit_reason: Optional[str] = None

    # Bearish (short-side) mirror — used by the two-sided screen (credit engine / Phase 4)
    bearish_technical_score: float = 0.0   # high when the DOWNtrend is strong
    bearish_entry_signal: bool = False     # mirror of entry_signal for downtrends


def compute_technical_signals(symbol: str, df: pd.DataFrame) -> Optional[TechnicalSignals]:
    """
    Compute all technical indicators from a daily OHLCV DataFrame.
    Returns None if insufficient data.
    """
    cfg = config.strategy

    if df is None or len(df) < 55:  # Need 55+ bars for EMA-50
        logger.warning(f"{symbol}: Insufficient data ({len(df) if df is not None else 0} bars), need 55+")
        return None

    try:
        df = df.copy()

        # ---- RSI ----
        rsi_series = ta.momentum.RSIIndicator(df["close"], window=cfg.rsi_period).rsi()
        current_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not pd.isna(rsi_series.iloc[-1]) else None

        # ---- MACD ----
        macd_indicator = ta.trend.MACD(df["close"], window_slow=cfg.macd_slow, window_fast=cfg.macd_fast, window_sign=cfg.macd_signal)
        macd_val = float(macd_indicator.macd().iloc[-1]) if macd_indicator is not None and not pd.isna(macd_indicator.macd().iloc[-1]) else None
        macd_sig = float(macd_indicator.macd_signal().iloc[-1]) if macd_indicator is not None and not pd.isna(macd_indicator.macd_signal().iloc[-1]) else None
        macd_hist_val = float(macd_indicator.macd_diff().iloc[-1]) if macd_indicator is not None and not pd.isna(macd_indicator.macd_diff().iloc[-1]) else None

        # Bullish crossover: MACD crossed above signal in last 7 bars (extended from 3)
        macd_crossover = False
        macd_cross_down = False
        if macd_indicator is not None:
            macd_diff_series = macd_indicator.macd_diff()
            if len(macd_diff_series) >= 8:
                recent_hist = macd_diff_series.iloc[-8:]
                if any(recent_hist.iloc[i] < 0 and recent_hist.iloc[i + 1] > 0 for i in range(len(recent_hist) - 1)):
                    macd_crossover = True
                # Bearish mirror: crossed from above-signal to below-signal
                if any(recent_hist.iloc[i] > 0 and recent_hist.iloc[i + 1] < 0 for i in range(len(recent_hist) - 1)):
                    macd_cross_down = True

        macd_bullish = bool(macd_val and macd_sig and macd_val > macd_sig)
        macd_bearish = bool(macd_val and macd_sig and macd_val < macd_sig)

        # ---- EMAs ----
        ema_9 = float(ta.trend.EMAIndicator(df["close"], window=9).ema_indicator().iloc[-1])
        ema_21 = float(ta.trend.EMAIndicator(df["close"], window=21).ema_indicator().iloc[-1])
        ema_50 = float(ta.trend.EMAIndicator(df["close"], window=50).ema_indicator().iloc[-1])
        close = float(df["close"].iloc[-1])

        # ---- Bollinger Bands ----
        bb_indicator = ta.volatility.BollingerBands(df["close"], window=cfg.bollinger_period, window_dev=cfg.bollinger_std)
        bb_upper = float(bb_indicator.bollinger_hband().iloc[-1]) if bb_indicator is not None and not pd.isna(bb_indicator.bollinger_hband().iloc[-1]) else None
        bb_mid = float(bb_indicator.bollinger_mavg().iloc[-1]) if bb_indicator is not None and not pd.isna(bb_indicator.bollinger_mavg().iloc[-1]) else None
        bb_lower = float(bb_indicator.bollinger_lband().iloc[-1]) if bb_indicator is not None and not pd.isna(bb_indicator.bollinger_lband().iloc[-1]) else None
        bb_pct = None
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_pct = (close - bb_lower) / (bb_upper - bb_lower)

        # ---- ATR ----
        atr_indicator = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=cfg.atr_period)
        atr_val = float(atr_indicator.average_true_range().iloc[-1]) if atr_indicator is not None and not pd.isna(atr_indicator.average_true_range().iloc[-1]) else None
        atr_pct = (atr_val / close * 100) if atr_val and close else None

        # ---- Volume ----
        vol_avg_20 = float(df["volume"].rolling(cfg.volume_ma_period).mean().iloc[-1])
        vol_current = float(df["volume"].iloc[-1])
        vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 0

        # ---- Stochastic ----
        stoch_indicator = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        stoch_k = float(stoch_indicator.stoch().iloc[-1]) if stoch_indicator is not None and not pd.isna(stoch_indicator.stoch().iloc[-1]) else None
        stoch_d = float(stoch_indicator.stoch_signal().iloc[-1]) if stoch_indicator is not None and not pd.isna(stoch_indicator.stoch_signal().iloc[-1]) else None
        stoch_bullish = bool(stoch_k and stoch_d and stoch_k > stoch_d and stoch_k < 80)

        # ============================================================
        # COMPOSITE TECHNICAL SCORE (0-1)
        # ============================================================
        score_components = []

        # RSI score (best between 40-60, penalty for extremes)
        if current_rsi:
            if cfg.rsi_min <= current_rsi <= cfg.rsi_max:
                rsi_score = 1.0 - abs(current_rsi - 50) / 25
            else:
                rsi_score = 0.0
            score_components.append(("rsi", rsi_score, 0.25))

        # MACD score — crossover gets full credit, bullish (above signal) gets partial
        macd_score = 0.0
        if macd_crossover:
            macd_score = 1.0
        elif macd_bullish:
            macd_score = 0.6
        score_components.append(("macd", macd_score, 0.30))

        # EMA trend score
        ema_score = 0.0
        if close > ema_50:
            ema_score += 0.5
        if close > ema_21 and close > ema_9:
            ema_score += 0.3
        if ema_9 > ema_21:
            ema_score += 0.2
        score_components.append(("ema", min(ema_score, 1.0), 0.25))

        # Volume score
        vol_score = min(vol_ratio / 1.5, 1.0)
        score_components.append(("volume", vol_score, 0.10))

        # Bollinger score (near lower band = potential bounce)
        bb_score = 0.0
        if bb_pct is not None:
            if 0.2 <= bb_pct <= 0.6:
                bb_score = 1.0 - abs(bb_pct - 0.4) * 2
        score_components.append(("bollinger", bb_score, 0.10))

        # Weighted sum
        total_weight = sum(w for _, _, w in score_components)
        technical_score = sum(s * w for _, s, w in score_components) / total_weight

        # ============================================================
        # BEARISH (SHORT-SIDE) MIRROR SCORE (0-1) — high when DOWNtrend is strong.
        # Same component weights, each criterion flipped. Used by the two-sided screen.
        # ============================================================
        bear_components = []
        # RSI: same "clean trend near mid, not at an extreme" shape, valid short band 22–70
        if current_rsi:
            if (100 - cfg.rsi_max) <= current_rsi <= (100 - cfg.rsi_min):
                bear_rsi = 1.0 - abs(current_rsi - 50) / 25
            else:
                bear_rsi = 0.0
            bear_components.append(("rsi", max(0.0, bear_rsi), 0.25))
        # MACD: bearish crossover full credit, below-signal partial
        bear_macd = 1.0 if macd_cross_down else (0.6 if macd_bearish else 0.0)
        bear_components.append(("macd", bear_macd, 0.30))
        # EMA: mirror — below the averages, fast under slow
        bear_ema = 0.0
        if close < ema_50:
            bear_ema += 0.5
        if close < ema_21 and close < ema_9:
            bear_ema += 0.3
        if ema_9 < ema_21:
            bear_ema += 0.2
        bear_components.append(("ema", min(bear_ema, 1.0), 0.25))
        # Volume: symmetric (conviction either way)
        bear_components.append(("volume", min(vol_ratio / 1.5, 1.0), 0.10))
        # Bollinger: near the UPPER band = rejection/short setup (mirror of lower-band bounce)
        bear_bb = 0.0
        if bb_pct is not None and 0.4 <= bb_pct <= 0.8:
            bear_bb = 1.0 - abs(bb_pct - 0.6) * 2
        bear_components.append(("bollinger", max(0.0, bear_bb), 0.10))
        bearish_technical_score = sum(s * w for _, s, w in bear_components) / total_weight

        # ============================================================
        # ENTRY SIGNAL
        # Daily entry_signal now requires MACD bullish (above signal line),
        # not a fresh crossover. Hourly MACD is checked separately for timing.
        # ============================================================
        # Daily entry_signal uses only slow, stale-tolerant indicators (RSI, EMA, volume).
        # MACD is intentionally excluded here — daily bars are 1 day stale during trading hours.
        # MACD timing is checked separately via hourly bars in scheduler.py.
        entry_signal = (
            cfg.rsi_min <= (current_rsi or 0) <= cfg.rsi_max and
            (close > ema_50 if cfg.require_above_ema_trend else True) and
            (vol_ratio >= cfg.volume_multiplier if cfg.require_volume_confirmation else True)
        )

        # Bearish entry — mirror: RSI in the short band (not yet oversold), price below the
        # 50-EMA (downtrend), volume confirmation if required.
        bearish_entry_signal = (
            (100 - cfg.rsi_max) <= (current_rsi or 100) <= (100 - cfg.rsi_min) and
            (close < ema_50 if cfg.require_above_ema_trend else True) and
            (vol_ratio >= cfg.volume_multiplier if cfg.require_volume_confirmation else True)
        )

        exit_reason = None
        exit_signal = False
        if current_rsi and current_rsi >= cfg.rsi_exit:
            exit_signal = True
            exit_reason = "rsi_overbought"
        elif macd_hist_val and macd_hist_val < -0.1 and macd_val and macd_val < macd_sig:
            exit_signal = True
            exit_reason = "macd_bearish"

        return TechnicalSignals(
            symbol=symbol,
            close=close,
            open=float(df["open"].iloc[-1]),
            high=float(df["high"].iloc[-1]),
            low=float(df["low"].iloc[-1]),
            volume=vol_current,
            volume_avg_20=vol_avg_20,
            volume_ratio=round(vol_ratio, 2),
            ema_9=round(ema_9, 2),
            ema_21=round(ema_21, 2),
            ema_50=round(ema_50, 2),
            above_ema_9_21=bool(close > ema_9 and close > ema_21),
            above_ema_50=bool(close > ema_50),
            ema_9_above_21=bool(ema_9 > ema_21),
            rsi=round(current_rsi, 2) if current_rsi else None,
            rsi_signal="oversold" if current_rsi and current_rsi < 30 else
                       "overbought" if current_rsi and current_rsi > 72 else "neutral",
            macd=round(macd_val, 4) if macd_val else None,
            macd_signal_line=round(macd_sig, 4) if macd_sig else None,
            macd_hist=round(macd_hist_val, 4) if macd_hist_val else None,
            macd_crossover=macd_crossover,
            macd_bullish=macd_bullish,
            macd_cross_down=macd_cross_down,
            macd_bearish=macd_bearish,
            atr=round(atr_val, 2) if atr_val else None,
            atr_pct=round(atr_pct, 2) if atr_pct else None,
            bollinger_upper=round(bb_upper, 2) if bb_upper else None,
            bollinger_lower=round(bb_lower, 2) if bb_lower else None,
            bollinger_mid=round(bb_mid, 2) if bb_mid else None,
            bollinger_pct=round(bb_pct, 3) if bb_pct else None,
            near_lower_band=bool(bb_pct and bb_pct < 0.25),
            stoch_k=round(stoch_k, 2) if stoch_k else None,
            stoch_d=round(stoch_d, 2) if stoch_d else None,
            stoch_bullish=stoch_bullish,
            technical_score=round(technical_score, 3),
            entry_signal=entry_signal,
            exit_signal=exit_signal,
            exit_reason=exit_reason,
            bearish_technical_score=round(bearish_technical_score, 3),
            bearish_entry_signal=bearish_entry_signal,
        )

    except Exception as e:
        logger.error(f"{symbol}: Technical signal computation failed: {e}")
        return None


def check_hourly_macd(df: pd.DataFrame) -> Tuple[bool, float, float]:
    """
    Compute MACD on hourly bars and return (is_bullish, macd_val, signal_val).
    Used as the second-level entry gate for precise timing.
    Fails open (returns True) if data is insufficient — don't block on data issues.
    """
    try:
        if df is None or len(df) < 35:
            return True, 0.0, 0.0
        macd_ind = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        h_macd = float(macd_ind.macd().iloc[-1])
        h_sig = float(macd_ind.macd_signal().iloc[-1])
        return h_macd > h_sig, round(h_macd, 4), round(h_sig, 4)
    except Exception:
        return True, 0.0, 0.0
