"""
Unit tests for the credit-spread engine (MASTER_STRATEGY.md CORE).
Pure logic — no network, no orders. Run: python scripts/test_credit_spreads.py

Covers:
  1. IV-environment classification (sell-rich / buy-cheap / middle)
  2. _select_strategy routing (legacy vs credit engine)
  3. check_credit_spread_entry — max-loss sizing, quality gate, cash reserve
  4. check_credit_spread_exit — 50% profit, 2x stop, 21-DTE, thesis break
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import config
from agent.risk import RiskEngine
from agent.signals.options_flow import (
    OptionsFlowSignals, _classify_iv_environment, _select_strategy,
)

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# ── 1. IV environment classification ──────────────────────────────────
print("\n[1] IV environment classification")
opts = config.options


def env(ivr, ratio):
    s = OptionsFlowSignals(symbol="X", iv_rank=ivr, iv_hv_ratio=ratio)
    return _classify_iv_environment(s, opts)


check("rich: IVR 60 + IV/HV 1.4", env(60, 1.4) == "rich")
check("NOT rich on IVR alone (high IVR, IV/HV 1.0)", env(60, 1.0) == "middle")
check("NOT rich when ratio unknown (avoid selling cheap vol)", env(60, None) == "middle")
check("cheap: IVR 15 + IV/HV 0.8", env(15, 0.8) == "cheap")
check("cheap allowed with unknown ratio (low IVR)", env(15, None) == "cheap")
check("middle: IVR 40 + IV/HV 1.1", env(40, 1.1) == "middle")
check("unknown when no IVR", env(None, 1.5) == "unknown")
# Loosened thresholds + single-signal overrides (2026-06-10 tuning)
check("rich via primary loosened (IVR 46 + IV/HV 1.16)", env(46, 1.16) == "rich")
check("rich via IVR override alone (IVR 85 + IV/HV 1.0)", env(85, 1.0) == "rich")
check("rich via ratio override alone (IVR 48 + IV/HV 1.40, SPY-like)", env(48, 1.40) == "rich")
check("still NOT rich: IVR 70 + IV/HV 1.05 (TSLA-like, no edge)", env(70, 1.05) == "middle")


# ── 2. _select_strategy routing ───────────────────────────────────────
print("\n[2] _select_strategy routing")


def mode(ivr, ratio, tech, sent, credit_on, condor_on=False, direction=None):
    saved_c, saved_ic = opts.enable_credit_spreads, opts.enable_iron_condor
    opts.enable_credit_spreads = credit_on
    opts.enable_iron_condor = condor_on
    try:
        s = OptionsFlowSignals(symbol="X", iv_rank=ivr, iv_hv_ratio=ratio)
        s.iv_regime = ("cheap" if (ivr or 0) < opts.iv_rank_cheap
                       else "expensive" if (ivr or 0) > opts.iv_rank_expensive else "normal")
        s.iv_environment = _classify_iv_environment(s, opts)
        return _select_strategy(s, tech, sent, direction)
    finally:
        opts.enable_credit_spreads, opts.enable_iron_condor = saved_c, saved_ic


# Credit engine ON — legacy score-inference (no explicit direction)
check("bullish + rich IV -> SELL_PUT_SPREAD", mode(60, 1.4, 0.70, 0.3, True) == "SELL_PUT_SPREAD")
check("bearish + rich IV -> SELL_CALL_SPREAD", mode(60, 1.4, 0.20, -0.4, True) == "SELL_CALL_SPREAD")
check("neutral + rich IV + condor off -> SKIP", mode(60, 1.4, 0.45, 0.0, True) == "SKIP")
check("neutral + rich IV + condor on -> IRON_CONDOR", mode(60, 1.4, 0.45, 0.0, True, True) == "IRON_CONDOR")
check("bullish + cheap IV -> BUY_CALL", mode(15, 0.8, 0.70, 0.3, True) == "BUY_CALL")
check("bullish + middle IV -> SKIP (no vol edge)", mode(40, 1.1, 0.70, 0.3, True) == "SKIP")

# Explicit two-sided direction (authoritative over scores)
check("dir=down + rich IV -> SELL_CALL_SPREAD", mode(60, 1.4, 0.5, 0.0, True, direction="down") == "SELL_CALL_SPREAD")
check("dir=down + cheap IV -> BUY_PUT", mode(15, 0.8, 0.5, 0.0, True, direction="down") == "BUY_PUT")
check("dir=up + rich IV -> SELL_PUT_SPREAD", mode(60, 1.4, 0.5, 0.0, True, direction="up") == "SELL_PUT_SPREAD")
check("dir=up + cheap IV -> BUY_CALL", mode(15, 0.8, 0.5, 0.0, True, direction="up") == "BUY_CALL")

# Legacy (credit engine OFF) — unchanged behavior
check("legacy bullish + normal IV -> BUY_CALL", mode(40, 1.1, 0.70, 0.3, False) == "BUY_CALL")
check("legacy bullish + expensive IV -> SKIP", mode(60, 1.4, 0.70, 0.3, False) == "SKIP")


# ── 3. check_credit_spread_entry — sizing ─────────────────────────────
print("\n[3] check_credit_spread_entry (max-loss sizing)")
risk = RiskEngine()
EQ, CASH = 25000.0, 25000.0  # budget = 2% = $500; reserve 20% = $5000


def entry(credit, width, cash=CASH, eq=EQ):
    return risk.check_credit_spread_entry("X", eq, cash, 0, credit, width)


r = entry(40, 100)  # max loss 60, credit/width 0.40
check("approved good spread", r.approved)
check("sized 8 contracts (500//60)", r.contracts == 8)
check("max_loss_per_spread = 60", r.max_loss_per_spread == 60)
check("total_max_loss = 480", r.total_max_loss == 480)

check("reject poor credit/width (0.15 < 0.20 gate)", not entry(15, 100).approved)
check("reject one spread over budget (maxloss 550 > 500, c/w 0.21 ok)", not entry(150, 700).approved)
check("reject when cash reserve leaves nothing", not entry(40, 100, cash=1000).approved)
check("reject invalid data", not entry(0, 100).approved)


# ── 4. check_credit_spread_exit — inverted math ───────────────────────
print("\n[4] check_credit_spread_exit (profit/stop/time/thesis)")
CR = 400.0  # credit received total


def ex(cost, dte=35, thesis=False):
    return risk.check_credit_spread_exit("X", CR, cost, dte, thesis_broken_hard=thesis)


check("take profit at 50% (cost 200 <= 200)", ex(200).should_exit and "credit_profit" in ex(200).reason)
check("take profit deeper (cost 120)", ex(120).should_exit)
check("hold mid-trade (cost 300, 30 DTE)", not ex(300, 30).should_exit)
check("stop at 2x credit (cost 800 >= 800)", ex(800).should_exit and "credit_stop" in ex(800).reason)
check("time exit at 21 DTE", ex(300, 21).should_exit and "time_exit" in ex(300, 21).reason)
check("thesis break cuts early", ex(300, 30, thesis=True).should_exit)
check("bad credit data -> no exit", not risk.check_credit_spread_exit("X", 0, 100, 30).should_exit)


# ── 5. Profit-lock exit for long options (SPY giveback fix) ───────────
print("\n[5] profit-lock exit (long options)")
plk = risk.check_option_exit("X", premium_paid=1000, current_value=2100, dte_remaining=30,
                             peak_value=2100, days_held=4)
check("locks profit at +100% gain", plk.should_exit and "profit_lock" in plk.reason)
nolk = risk.check_option_exit("X", premium_paid=1000, current_value=1400, dte_remaining=30,
                              peak_value=1400, days_held=4)
check("does NOT lock at +40% (let it run)", not (nolk.should_exit and "profit_lock" in (nolk.reason or "")))


# ── 6. Two-sided technical screen (bearish mirror) ────────────────────
print("\n[6] technical bearish screen")
import numpy as np
import pandas as pd
from agent.signals.technical import compute_technical_signals


def make_df(trend):
    # gentle trend + oscillation so RSI stays mid-range (not pinned at an extreme)
    n = 80
    base = 100.0
    closes = [base + trend * i + 4.0 * np.sin(i / 4.0) for i in range(n)]
    closes = np.array(closes)
    highs = closes + 1.0
    lows = closes - 1.0
    opens = closes - trend
    vol = np.full(n, 1_000_000.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx)


up = compute_technical_signals("UP", make_df(+0.4))
down = compute_technical_signals("DN", make_df(-0.4))
# Cross-dataset: bull scorer should favor the uptrend; bear scorer should favor the downtrend.
check("bull scorer favors uptrend (up > down)", up.technical_score > down.technical_score)
check("bear scorer favors downtrend (down > up)", down.bearish_technical_score > up.bearish_technical_score)
check("uptrend: price above 50-EMA", up.above_ema_50)
check("downtrend: price below 50-EMA", not down.above_ema_50)
check("bearish entry signal types as bool", isinstance(down.bearish_entry_signal, bool))


# ── 7. Reversal / exhaustion module ───────────────────────────────────
print("\n[7] reversal/exhaustion signals")
from agent.signals.reversal import compute_reversal_signals

# Construct a bullish divergence: price makes a lower low while it turns up on the last bar.
n = 70
c = [100 - 0.5 * i for i in range(40)] + [80 + 0.3 * (i - 40) for i in range(40, n)]
c = np.array(c, dtype=float)
df_rev = pd.DataFrame({
    "open": c, "high": c + 1, "low": c - 1, "close": c,
    "volume": np.concatenate([np.full(60, 1e6), np.full(10, 3e6)]),
}, index=pd.date_range("2026-01-01", periods=n, freq="D"))
rev = compute_reversal_signals("REV", df_rev)
check("reversal returns a ReversalSignals", hasattr(rev, "reversal_up") and hasattr(rev, "reversal_down"))
check("reversal flags are booleans", isinstance(rev.reversal_up, bool) and isinstance(rev.reversal_down, bool))
check("fib levels computed", isinstance(rev.fib_levels, dict))
check("does not crash on short/empty data", compute_reversal_signals("E", df_rev.head(5)) is not None)


# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*48}\nRESULT: {passed} passed, {failed} failed\n{'='*48}")
sys.exit(1 if failed else 0)
