"""
Trend-Buyer v1 — Entry filters.

Turns a raw 9/21 EMA cross into a high-quality, confirmed-trend signal. A trade is
allowed ONLY if every enabled filter passes for the signal's direction:

  1. ADX > min            — a real trend exists (not chop / whipsaw)
  2. Regime               — price on the right side of a rising/falling 200-SMA, 50/200 aligned
  3. Anti-chase           — price isn't already over-extended (don't buy after the move)
  4. Volume confirmation  — the latest bar trades above its recent average
  5. Cheap vol            — vol-rank is low (we don't overpay implied volatility)

`evaluate(df, direction)` returns a FilterResult with the verdict, a human reason,
and the computed metrics (also stored on the trade / logged).
"""
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

from agent.config import config
from agent.signals import indicators as ind


@dataclass
class FilterResult:
    passed: bool
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


def evaluate(df: pd.DataFrame, direction: str, cfg=None) -> FilterResult:
    """direction: 'bull' (candidate CALL) or 'bear' (candidate PUT). df = closed daily bars.
    `cfg` overrides the global config (used by the backtester's parameter sweeps); defaults
    to the live singleton so production behavior is unchanged."""
    cfg = cfg or config
    f = cfg.filters
    ivc = cfg.iv
    metrics: Dict[str, float] = {}

    if df is None or len(df) < max(f.regime_trend_sma + f.regime_slope_lookback, 60):
        return FilterResult(False, "insufficient history for filters", metrics)

    close = df["close"].astype(float)
    price = float(close.iloc[-1])
    metrics["price"] = round(price, 2)

    # --- 1) ADX trend strength ---
    adx_val = ind.adx(df, f.adx_period)
    metrics["adx"] = round(adx_val, 1) if adx_val is not None else None
    if f.require_adx:
        if adx_val is None:
            return FilterResult(False, "ADX unavailable", metrics)
        if adx_val < f.adx_min:
            return FilterResult(False, f"ADX {adx_val:.1f} < {f.adx_min} (no trend / choppy)", metrics)

    # --- ATR (used by anti-chase + carried for the hard stop) ---
    atr_val = ind.atr(df, f.atr_period)
    metrics["atr"] = round(atr_val, 2) if atr_val is not None else None

    # --- 2) Regime ---
    sma_fast = ind.sma(close, f.regime_fast_sma)
    sma_slow = ind.sma(close, f.regime_trend_sma)
    sf, ss = float(sma_fast.iloc[-1]), float(sma_slow.iloc[-1])
    ss_prior = float(sma_slow.iloc[-1 - f.regime_slope_lookback])
    metrics["sma50"], metrics["sma200"] = round(sf, 2), round(ss, 2)
    if f.require_regime:
        if direction == "bull":
            if not (price > ss and sf > ss and ss > ss_prior):
                return FilterResult(False,
                    f"regime not bullish (px {price:.2f} vs 200SMA {ss:.2f}, 50>{200} {sf > ss}, rising {ss > ss_prior})",
                    metrics)
        else:  # bear
            if not (price < ss and sf < ss and ss < ss_prior):
                return FilterResult(False,
                    f"regime not bearish (px {price:.2f} vs 200SMA {ss:.2f}, 50<{200} {sf < ss}, falling {ss < ss_prior})",
                    metrics)

    # --- 3) Anti-chase: extension from the reference SMA, in ATR units ---
    if f.anti_chase and atr_val:
        ref = float(ind.sma(close, f.chase_ref_sma).iloc[-1])
        ext_atr = (price - ref) / atr_val
        metrics["extension_atr"] = round(ext_atr, 2)
        if direction == "bull" and ext_atr > f.chase_atr_max:
            return FilterResult(False, f"chasing: {ext_atr:.2f} ATR above 50SMA (> {f.chase_atr_max})", metrics)
        if direction == "bear" and ext_atr < -f.chase_atr_max:
            return FilterResult(False, f"chasing: {ext_atr:.2f} ATR below 50SMA (< -{f.chase_atr_max})", metrics)

    # --- 4) Volume confirmation ---
    if f.require_volume and "volume" in df.columns:
        vol = df["volume"].astype(float)
        avg_vol = float(vol.iloc[-f.volume_lookback:].mean())
        last_vol = float(vol.iloc[-1])
        metrics["vol_x_avg"] = round(last_vol / avg_vol, 2) if avg_vol else None
        if avg_vol and last_vol < avg_vol:
            return FilterResult(False, f"weak volume ({last_vol/avg_vol:.2f}x avg)", metrics)

    # --- 5) Cheap vol gate ---
    vr = ind.vol_rank(close, ivc.hv_window, ivc.vol_rank_lookback)
    metrics["vol_rank"] = vr
    if ivc.require_cheap_vol:
        if vr is None:
            return FilterResult(False, "vol-rank unavailable", metrics)
        if vr > ivc.vol_rank_max:
            return FilterResult(False, f"vol too rich (vol-rank {vr:.0f} > {ivc.vol_rank_max:.0f})", metrics)

    return FilterResult(True, "all filters passed", metrics)
