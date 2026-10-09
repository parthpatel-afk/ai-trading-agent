"""
Synthesize + reprice a deep-ITM option for the backtester.

The live bot picks a real contract off the chain by Black-Scholes delta; the backtester has
no historical chain, so it does the mirror image: given a target delta it INVERTS Black-Scholes
to a strike (agent.pricing.strike_for_delta), snaps to a realistic listed-strike grid, picks a
standard monthly expiration (~target DTE), and prices with Black-Scholes. Each later day it
reprices from the underlying's close. Deep-ITM ⇒ price ≈ intrinsic ⇒ this is accurate (proven
in validate.py).

Volatility input (`sigma`): the strategy has no historical implied vol, so we use the
underlying's HISTORICAL volatility (HV) — the same measure the entry's cheap-vol gate uses —
as the vol for both strike selection and pricing (consistent by construction). We hold the
entry sigma fixed over the trade's life (deep-ITM vega is small, and it isolates P&L to the
underlying's move); the model still updates price via time decay and the changing spot.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd

from agent import pricing
from agent.config import config
from agent.executor import _target_monthly_date  # pure date math; reuse the live expiry picker


def hv_sigma(closes: pd.Series, window: Optional[int] = None, cfg=None) -> Optional[float]:
    """Annualized historical volatility from daily closes (the vol input for pricing).
    Uses the same HV window the entry's vol-rank gate uses (config.iv.hv_window)."""
    cfg = cfg or config
    window = window or cfg.iv.hv_window
    if closes is None or len(closes) < window + 2:
        return None
    log_ret = np.log(closes.astype(float) / closes.astype(float).shift(1))
    hv = log_ret.rolling(window).std() * np.sqrt(252)
    v = hv.iloc[-1]
    return float(v) if pd.notna(v) and v > 0 else None


@dataclass
class SynthContract:
    underlying: str
    option_type: str          # "CALL" | "PUT"
    strike: float
    expiration: date
    dte_at_entry: int
    entry_sigma: float        # annualized HV used for pricing (held fixed over the trade)
    entry_price: float        # per share (× 100 = one contract)
    entry_spot: float
    entry_delta: float        # realized |delta| of the chosen strike (≈ target)


def pick_expiration(asof: date, cfg=None) -> Optional[date]:
    """Standard monthly (3rd Friday) closest to target DTE within [min_dte, max_dte], as of `asof`."""
    o = (cfg or config).options
    return _target_monthly_date(o.target_dte, o.min_dte, o.max_dte, asof)


def select_contract(underlying: str, option_type: str, spot: float, sigma: float,
                    asof: date, target_delta: float, cfg=None) -> Optional[SynthContract]:
    """Build the deep-ITM contract at entry. Returns None if inputs unusable / no expiry fits."""
    o = (cfg or config).options
    if spot <= 0 or sigma is None or sigma <= 0:
        return None
    exp = pick_expiration(asof, cfg)
    if exp is None:
        return None
    dte = (exp - asof).days
    strike = pricing.strike_for_delta(spot, target_delta, dte, sigma, option_type, o.risk_free_rate)
    if not strike or strike <= 0:
        return None
    px = pricing.bs_price(spot, strike, dte, sigma, option_type, o.risk_free_rate)
    if px is None or px <= 0:
        return None
    realized_delta = pricing.bs_delta(spot, strike, dte, sigma, option_type, o.risk_free_rate) or target_delta
    return SynthContract(underlying, option_type, round(strike, 2), exp, dte,
                         round(sigma, 4), round(px, 4), round(spot, 4), round(realized_delta, 4))


def reprice(contract: SynthContract, spot: float, asof: date,
            sigma: Optional[float] = None, cfg=None) -> float:
    """Per-share option value on `asof` given the underlying's `spot`. At/after expiry →
    pure intrinsic. Uses the entry sigma unless a fresh one is supplied."""
    o = (cfg or config).options
    dte = max(0, (contract.expiration - asof).days)
    s = sigma if (sigma is not None and sigma > 0) else contract.entry_sigma
    px = pricing.bs_price(spot, contract.strike, dte, s, contract.option_type, o.risk_free_rate)
    if px is None:
        return pricing.intrinsic_value(spot, contract.strike, contract.option_type)
    return px
