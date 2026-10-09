"""
Trend-Buyer v1 — Strategy helpers: signal dedup, contract depth, sizing, and exit math.

Pure functions (no I/O) so they're unit-testable. The scheduler owns the state machine
(peaks, confirmation counts, partial-done flags) and calls these for the decisions.
"""
from dataclasses import dataclass
from typing import Optional

from agent.config import config


# ── Entry decision (EMA fresh-cross dedup; filters are applied separately) ──────────

@dataclass
class TradeDecision:
    action: str                        # "BUY_CALL" | "BUY_PUT" | "HOLD"
    option_type: Optional[str] = None  # "CALL" | "PUT" | None
    reason: str = ""


def decide(ema_state, current_side: Optional[str]) -> TradeDecision:
    """Act on a FRESH crossover exactly once. The opposite cross flips us; nothing else
    triggers an entry. `current_side` = "CALL" | "PUT" | None (what we already hold)."""
    if ema_state is None:
        return TradeDecision("HOLD", None, "no EMA state")
    if not ema_state.fresh or ema_state.cross is None:
        return TradeDecision("HOLD", None,
                             f"no fresh cross (last={ema_state.cross}, age={ema_state.cross_age_days}d)")
    desired = "CALL" if ema_state.cross == "BULL" else "PUT"
    if desired == current_side:
        return TradeDecision("HOLD", desired, f"already holding {desired} for {ema_state.cross} cross")
    action = "BUY_CALL" if desired == "CALL" else "BUY_PUT"
    return TradeDecision(action, desired, f"fresh {ema_state.cross} cross {ema_state.cross_age_days}d ago")


# ── Contract depth (decision #1: deeper ITM when vol is expensive) ──────────────────

def target_delta_for(vol_rank: Optional[float], cfg=None) -> float:
    """Default deep-ITM target, stepped deeper when vol is expensive so there's almost no
    extrinsic value to overpay for. `cfg` overrides the global (backtester sweeps)."""
    o = (cfg or config).options
    if vol_rank is not None and vol_rank > o.deep_itm_vol_threshold:
        return o.deep_itm_target_delta
    return o.target_delta


# ── Sizing (on the STOP distance, with capital + risk caps) ─────────────────────────

@dataclass
class SizeResult:
    contracts: int
    reason: str
    est_stop_loss_per_contract: float = 0.0


def estimate_stop_loss_per_contract(delta: float, atr: float, stop_atr_mult: float) -> float:
    """Approx $ lost per contract if the hard stop hits: Δoption ≈ delta × Δunderlying × 100,
    with Δunderlying = stop_atr_mult × ATR."""
    return abs(delta) * (stop_atr_mult * atr) * 100.0


def size_position(equity: float, cash: float, ask_per_contract: float,
                  est_stop_loss_per_contract: float, cfg=None) -> SizeResult:
    """Contracts = the most conservative of: risk cap, capital cap, and cash — flooring to int.
    A single chunky deep-ITM contract is allowed through only if it clears the capital cap,
    cash, and an absolute risk ceiling; otherwise the trade is REJECTED (never force-size).
    `cfg` overrides the global (backtester sweeps)."""
    a = (cfg or config).account
    if ask_per_contract <= 0:
        return SizeResult(0, "no ask price")

    risk_budget = equity * a.risk_per_trade_pct
    capital_cap = equity * a.max_position_pct
    cash_avail = max(0.0, cash - equity * a.cash_reserve_pct)

    by_risk = int(risk_budget // est_stop_loss_per_contract) if est_stop_loss_per_contract > 0 else 9999
    by_capital = int(capital_cap // ask_per_contract)
    by_cash = int(cash_avail // ask_per_contract)
    n = min(by_risk, by_capital, by_cash)

    if n < 1:
        # Allow exactly one contract if it's affordable, within the capital cap, and its
        # stop-loss is within the absolute risk ceiling. Else reject (don't oversize).
        within_risk = est_stop_loss_per_contract <= equity * a.max_risk_per_trade_pct
        if ask_per_contract <= capital_cap and ask_per_contract <= cash_avail and within_risk:
            return SizeResult(1, "1 contract (min size, within caps)", est_stop_loss_per_contract)
        why = []
        if ask_per_contract > capital_cap:
            why.append(f"premium ${ask_per_contract:.0f} > capital cap ${capital_cap:.0f}")
        if ask_per_contract > cash_avail:
            why.append(f"premium ${ask_per_contract:.0f} > cash avail ${cash_avail:.0f}")
        if not within_risk:
            why.append(f"stop-loss ${est_stop_loss_per_contract:.0f} > {a.max_risk_per_trade_pct:.1%} ceiling")
        return SizeResult(0, "; ".join(why) or "0 contracts")
    return SizeResult(n, f"{n} contract(s)", est_stop_loss_per_contract)


# ── Exit math (all on the UNDERLYING's daily close) ─────────────────────────────────

def hard_stop_level(entry_underlying: float, atr: float, direction: str, cfg=None) -> float:
    """Initial hard stop price on the underlying. CALL: below entry; PUT: above entry."""
    mult = (cfg or config).exits.hard_stop_atr_mult
    return entry_underlying - mult * atr if direction == "CALL" else entry_underlying + mult * atr


def chandelier_level(extreme_close: float, atr: float, direction: str, cfg=None) -> float:
    """Chandelier trail on the runner. CALL: highest_close − k·ATR; PUT: lowest_close + k·ATR."""
    mult = (cfg or config).exits.chandelier_atr_mult
    return extreme_close - mult * atr if direction == "CALL" else extreme_close + mult * atr


def apply_breakeven_floor(line: float, entry_underlying: float, direction: str) -> float:
    """After a partial is taken, never let the runner trail past ~breakeven on the underlying."""
    return max(line, entry_underlying) if direction == "CALL" else min(line, entry_underlying)


def stop_or_trail_breached(close: float, level: float, direction: str) -> bool:
    """A long CALL exits when price closes at/below the line; a PUT at/above it."""
    return close <= level if direction == "CALL" else close >= level
