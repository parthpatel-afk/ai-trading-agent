"""
Trend-Buyer — the exit decision, as ONE pure function.

`decide_exit(...)` is the single authority for "what should happen to an open position
today", judged on the UNDERLYING's daily close (never the option mark — the hard-won
ledger rule). It performs NO I/O: it takes the position state + today's readings and
returns an ExitDecision. Both callers act on that decision:

  • the LIVE bot  — `scheduler.run_close_monitor` executes the close/partial + persists streaks
  • the BACKTESTER — replays it over history to measure the exits we can't yet observe live

Extracting this guarantees the backtest tests the *real* exit logic, not a copy that drifts.
The layered exit it encodes (unchanged from the original monitor):
  1. Theta time-cap  — never hold into the decay red zone (fires first, regardless of price)
  2. Partial take    — bank profit at +target, which arms the trail (only while not yet armed)
  3. Hard stop       — cut losers: underlying beyond the stop, or closed across the 21-EMA,
                       or option premium collapsed (backstop); N confirming closes required
  4. Chandelier trail— on the runner once armed: exit on N closes beyond highest/lowest − k·ATR
"""
from dataclasses import dataclass
from typing import Optional

from agent.config import config
from agent import strategy


# Actions the scheduler / backtester switch on.
HOLD = "HOLD"                 # nothing to do; persist refreshed peak + streaks
TIME_EXIT = "TIME_EXIT"       # close full — theta time-cap
PARTIAL_ARM = "PARTIAL_ARM"   # take partial (or arm on 1 contract), arms the trail
HARD_STOP = "HARD_STOP"       # close full — initial hard stop hit (confirmed)
TRAIL_EXIT = "TRAIL_EXIT"     # close full — Chandelier trail hit (confirmed)
WAIT_STOP = "WAIT_STOP"       # stop breached but not yet confirmed — persist streak
WAIT_TRAIL = "WAIT_TRAIL"     # trail breached but not yet confirmed — persist streak


@dataclass
class PositionState:
    """The subset of an open trade the exit logic needs (decoupled from the DB model,
    so the backtester can build one directly)."""
    option_type: str                    # "CALL" | "PUT"
    entry_underlying: float
    hard_stop_price: float
    peak_close: Optional[float] = None  # best close since entry (high for calls, low for puts)
    trail_armed: bool = False
    stop_streak: int = 0
    trail_streak: int = 0

    @classmethod
    def from_trade(cls, t) -> "PositionState":
        return cls(
            option_type=t.option_type,
            entry_underlying=float(t.entry_underlying),
            hard_stop_price=float(t.hard_stop_price),
            peak_close=(float(t.peak_close) if t.peak_close is not None else None),
            trail_armed=bool(t.trail_armed),
            stop_streak=int(t.stop_streak or 0),
            trail_streak=int(t.trail_streak or 0),
        )


@dataclass
class ExitDecision:
    action: str
    reason: str
    new_peak: float
    new_stop_streak: int
    new_trail_streak: int
    trail_line: Optional[float] = None   # the trail level, for the WAIT_TRAIL log

    @property
    def closes_position(self) -> bool:
        return self.action in (TIME_EXIT, HARD_STOP, TRAIL_EXIT)


def decide_exit(pos: PositionState, close: Optional[float], chand_atr: float,
                ema21: Optional[float], option_gain_pct: float,
                dte_remaining: int, cfg=None) -> ExitDecision:
    """Decide today's action for one open position. Pure — no I/O.

    close/ema21 may be None (data unavailable): the time-cap still fires (as in the live
    monitor, which checks DTE before fetching price), otherwise we HOLD unchanged."""
    cfg = cfg or config
    ex = cfg.exits
    direction = pos.option_type
    cur_peak = pos.peak_close if pos.peak_close is not None else pos.entry_underlying
    stop_streak = int(pos.stop_streak or 0)
    trail_streak = int(pos.trail_streak or 0)

    # 1) Theta time-cap — checked FIRST, before any price math (fires even without a quote).
    if dte_remaining <= ex.time_exit_dte:
        return ExitDecision(TIME_EXIT, f"time_exit_{ex.time_exit_dte}dte",
                            cur_peak, stop_streak, trail_streak)

    # No usable close → no-op (mirrors the monitor's `continue`).
    if close is None:
        return ExitDecision(HOLD, "no close available", cur_peak, stop_streak, trail_streak)

    # Refresh the peak/trough close since entry.
    peak = max(cur_peak, close) if direction == "CALL" else min(cur_peak, close)

    if not pos.trail_armed:
        # 2) Partial profit-take (good news first) → arms the trail.
        if option_gain_pct >= ex.partial_profit_pct:
            return ExitDecision(PARTIAL_ARM,
                                f"+{ex.partial_profit_pct:.0%} option target", peak,
                                stop_streak, trail_streak)
        # 3) Hard stop: underlying beyond the stop, OR closed across the 21-EMA, OR premium collapsed.
        beyond = strategy.stop_or_trail_breached(close, pos.hard_stop_price, direction)
        ema_break = ex.use_ema_stop and ema21 is not None and (
            (direction == "CALL" and close < ema21) or (direction == "PUT" and close > ema21))
        prem_break = option_gain_pct <= -ex.premium_stop_pct
        if beyond or ema_break or prem_break:
            streak = stop_streak + 1
            if streak >= ex.stop_confirm_reads or prem_break:
                return ExitDecision(HARD_STOP, "hard_stop", peak, streak, trail_streak)
            return ExitDecision(WAIT_STOP,
                                f"hard-stop read {streak}/{ex.stop_confirm_reads}", peak,
                                streak, trail_streak)
        return ExitDecision(HOLD, "holding (pre-trail)", peak, 0, trail_streak)

    # 4) Chandelier trail on the runner (post-partial).
    line = strategy.chandelier_level(peak, chand_atr, direction, cfg)
    line = strategy.apply_breakeven_floor(line, pos.entry_underlying, direction)
    if strategy.stop_or_trail_breached(close, line, direction):
        streak = trail_streak + 1
        if streak >= ex.trail_confirm_reads:
            return ExitDecision(TRAIL_EXIT, "chandelier_trail", peak, stop_streak, streak, line)
        return ExitDecision(WAIT_TRAIL,
                            f"trail read {streak}/{ex.trail_confirm_reads}", peak,
                            stop_streak, streak, line)
    return ExitDecision(HOLD, "holding (trailing)", peak, stop_streak, 0, line)
