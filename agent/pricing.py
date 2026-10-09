"""
Trend-Buyer — Black-Scholes option math (shared, pure, no dependencies beyond `math`).

ONE source of truth for option pricing, used by BOTH:
  • the LIVE bot   — `executor` picks a strike by Black-Scholes delta, and
  • the BACKTESTER — which has no historical option chain, so it *synthesizes* the
                     deep-ITM contract and reprices it each day from the underlying.

Why we can price options from the stock alone (the backtest's key idea): an option's
value = intrinsic (how deep in-the-money = max(S-K,0) for a call) + a little time value.
Our strategy deliberately buys DEEP in-the-money (delta ~0.75-0.82), where intrinsic
dominates and the time-value sliver is small — exactly the regime where Black-Scholes,
fed a reasonable volatility, reconstructs the option price accurately. So we never need
to buy expensive historical option-price data to backtest THIS style honestly.

Conventions: spot/strike in dollars, `dte` in calendar days, `iv` annualized (0.30 = 30%),
`r` the annual risk-free rate, `option_type` in {"CALL","PUT"}. Deltas returned as a
MAGNITUDE in [0,1] (calls and puts both positive) to match how the strategy reasons about
"depth". Time is measured in calendar days / 365 to match the live executor exactly.
"""
import math
from typing import Optional

__all__ = [
    "norm_cdf", "norm_ppf", "bs_delta", "bs_price",
    "intrinsic_value", "strike_for_delta",
]


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (matches the live executor)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (probit) — Acklam's rational approximation,
    accurate to ~1e-9 over p in (0,1). Used to invert a target delta into a strike."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Coefficients for Acklam's algorithm.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _d1_d2(spot: float, strike: float, dte: int, iv: float, r: float):
    T = max(int(dte), 1) / 365.0
    vol = iv * math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / vol
    return d1, d1 - vol, T


def bs_delta(spot: float, strike: float, dte: int, iv: float,
             option_type: str, r: float = 0.04) -> Optional[float]:
    """Black-Scholes delta MAGNITUDE (0..1). None if inputs are unusable.
    ~0.75 == 'deep in-the-money'. Identical formula to the live executor's picker."""
    try:
        if spot <= 0 or strike <= 0 or iv <= 0 or int(dte) <= 0:
            return None
        d1, _, _ = _d1_d2(spot, strike, dte, iv, r)
        nd1 = norm_cdf(d1)
        return abs(nd1 if option_type == "CALL" else (nd1 - 1.0))
    except Exception:
        return None


def bs_price(spot: float, strike: float, dte: int, iv: float,
             option_type: str, r: float = 0.04) -> Optional[float]:
    """Black-Scholes fair value PER SHARE (× 100 for one contract). None if unusable.
    At/after expiry (dte<=0) collapses to pure intrinsic value."""
    try:
        if spot <= 0 or strike <= 0:
            return None
        if int(dte) <= 0 or iv <= 0:
            return intrinsic_value(spot, strike, option_type)
        d1, d2, T = _d1_d2(spot, strike, dte, iv, r)
        disc = math.exp(-r * T)
        if option_type == "CALL":
            px = spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
        else:
            px = strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)
        # Never return less than intrinsic (guards tiny numerical undershoot deep ITM).
        return max(px, intrinsic_value(spot, strike, option_type))
    except Exception:
        return None


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    """The guaranteed part: how deep in-the-money the option is (>= 0)."""
    return max(spot - strike, 0.0) if option_type == "CALL" else max(strike - spot, 0.0)


def strike_for_delta(spot: float, target_delta: float, dte: int, iv: float,
                     option_type: str, r: float = 0.04,
                     increment: Optional[float] = None) -> Optional[float]:
    """Invert Black-Scholes delta → the strike that gives ~target_delta depth.
    This is how the BACKTESTER synthesizes a deep-ITM contract when there is no chain.

    For a CALL, delta = N(d1) so d1 = norm_ppf(target); for a PUT, |delta| = N(-d1)
    so d1 = -norm_ppf(target). Then solve d1's definition for K. Optionally snap to a
    realistic strike ladder increment (e.g. $1 / $2.50 / $5)."""
    try:
        if spot <= 0 or iv <= 0 or int(dte) <= 0 or not (0.0 < target_delta < 1.0):
            return None
        T = max(int(dte), 1) / 365.0
        vol = iv * math.sqrt(T)
        d1 = norm_ppf(target_delta) if option_type == "CALL" else -norm_ppf(target_delta)
        # ln(S/K) = d1*vol - (r + 0.5 iv^2) T  →  K = S / exp(d1*vol - (r+0.5 iv^2)T)
        strike = spot / math.exp(d1 * vol - (r + 0.5 * iv * iv) * T)
        inc = increment if increment is not None else _strike_increment(spot)
        return round(round(strike / inc) * inc, 2) if inc else round(strike, 2)
    except Exception:
        return None


def _strike_increment(spot: float) -> float:
    """A realistic listed-strike spacing for a given price level (approx US equity ladders)."""
    if spot < 25:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 250:
        return 2.5
    return 5.0
