"""
Unit tests for the Phase-0 backtest foundation (no network):
  • agent.pricing        — Black-Scholes price/delta, delta→strike inversion
  • agent.exits          — the extracted pure exit decision (decide_exit)
  • agent.config         — clone_with() for parameter sweeps
  • injectable cfg       — filters/strategy honor an overridden config

Run: python scripts/test_backtest_foundation.py   (ASCII output; no UTF-8 console needed)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import pricing, exits, strategy
from agent.config import config
from agent.exits import PositionState, decide_exit

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

# ── Black-Scholes pricing ────────────────────────────────────────────
print("PRICING — delta, price, intrinsic")
S, r = 100.0, 0.04
# Deep-ITM call (strike 80) should have high delta; ATM ~0.5-0.6; OTM low.
d_itm = pricing.bs_delta(S, 80, 75, 0.30, "CALL", r)
d_atm = pricing.bs_delta(S, 100, 75, 0.30, "CALL", r)
d_otm = pricing.bs_delta(S, 120, 75, 0.30, "CALL", r)
check("deep-ITM call delta > 0.75", d_itm > 0.75)
check("delta ordering ITM>ATM>OTM", d_itm > d_atm > d_otm)
check("ATM call delta near 0.5-0.6", 0.5 < d_atm < 0.65)
# Price never below intrinsic; deep ITM ~ mostly intrinsic.
p = pricing.bs_price(S, 80, 75, 0.30, "CALL", r)
check("deep-ITM price >= intrinsic (20)", p >= 20.0)
check("deep-ITM price mostly intrinsic (<26)", p < 26.0)
check("expired price == intrinsic", abs(pricing.bs_price(S, 80, 0, 0.30, "CALL") - 20.0) < 1e-9)
check("OTM put intrinsic is 0", pricing.intrinsic_value(100, 80, "PUT") == 0.0)
# Put-call parity: C - P = S - K e^{-rT}
import math
T = 75 / 365.0
C = pricing.bs_price(S, 95, 75, 0.30, "CALL", r)
P = pricing.bs_price(S, 95, 75, 0.30, "PUT", r)
check("put-call parity holds", abs((C - P) - (S - 95 * math.exp(-r * T))) < 1e-6)

print("PRICING — norm_ppf inverts norm_cdf")
check("ppf(cdf(1.23)) ~ 1.23", abs(pricing.norm_ppf(pricing.norm_cdf(1.23)) - 1.23) < 1e-6)
check("cdf(0)=0.5", abs(pricing.norm_cdf(0.0) - 0.5) < 1e-12)

print("PRICING — strike_for_delta round-trips to the target delta")
for opt in ("CALL", "PUT"):
    k = pricing.strike_for_delta(S, 0.75, 75, 0.30, opt, r, increment=0)  # no snapping
    back = pricing.bs_delta(S, k, 75, 0.30, opt, r)
    check(f"{opt}: recovered delta ~ 0.75 (got {back:.3f})", abs(back - 0.75) < 0.01)
    check(f"{opt}: strike is ITM-side of spot",
          (k < S) if opt == "CALL" else (k > S))
ksnap = pricing.strike_for_delta(S, 0.75, 75, 0.30, "CALL", r)  # snapped ($2.50 @ $100)
check("snapped strike lands on a $2.50 grid", abs((ksnap / 2.5) - round(ksnap / 2.5)) < 1e-9)

# ── Exit decision (pure) ─────────────────────────────────────────────
def pos(**kw):
    base = dict(option_type="CALL", entry_underlying=100.0, hard_stop_price=96.0,
                peak_close=100.0, trail_armed=False, stop_streak=0, trail_streak=0)
    base.update(kw)
    return PositionState(**base)

ex = config.exits
print("EXITS — time cap fires first (even with no close)")
d = decide_exit(pos(), close=None, chand_atr=2.0, ema21=99.0, option_gain_pct=0.0,
                dte_remaining=ex.time_exit_dte)  # dte == cap -> exit
check("dte<=cap -> TIME_EXIT", d.action == exits.TIME_EXIT and d.closes_position)

print("EXITS — no close -> HOLD no-op")
d = decide_exit(pos(peak_close=105.0), close=None, chand_atr=2.0, ema21=99.0,
                option_gain_pct=0.0, dte_remaining=75)
check("close None -> HOLD", d.action == exits.HOLD)
check("HOLD keeps peak unchanged", d.new_peak == 105.0)

print("EXITS — partial take arms the trail")
d = decide_exit(pos(), close=112.0, chand_atr=2.0, ema21=99.0,
                option_gain_pct=ex.partial_profit_pct + 0.05, dte_remaining=75)
check("gain>=target -> PARTIAL_ARM", d.action == exits.PARTIAL_ARM)
check("peak advanced to new high", d.new_peak == 112.0)

print("EXITS — hard stop needs 2 confirming reads (CALL below stop)")
d1 = decide_exit(pos(stop_streak=0), close=95.0, chand_atr=2.0, ema21=99.0,
                 option_gain_pct=-0.1, dte_remaining=75)      # 95 <= stop 96
check("1st breach -> WAIT_STOP streak 1", d1.action == exits.WAIT_STOP and d1.new_stop_streak == 1)
d2 = decide_exit(pos(stop_streak=1), close=95.0, chand_atr=2.0, ema21=99.0,
                 option_gain_pct=-0.1, dte_remaining=75)
check("2nd breach -> HARD_STOP", d2.action == exits.HARD_STOP)

print("EXITS — premium collapse is an immediate stop (no wait)")
d = decide_exit(pos(stop_streak=0), close=99.0, chand_atr=2.0, ema21=99.5,
                option_gain_pct=-(ex.premium_stop_pct + 0.01), dte_remaining=75)
check("prem_break -> HARD_STOP at streak 0", d.action == exits.HARD_STOP)

print("EXITS — 21-EMA break triggers the stop path")
d = decide_exit(pos(stop_streak=1), close=98.0, chand_atr=2.0, ema21=99.0,  # close<ema, above price-stop 96
                option_gain_pct=-0.05, dte_remaining=75)
check("close below 21-EMA -> HARD_STOP (2nd read)", d.action == exits.HARD_STOP)

print("EXITS — healthy hold resets the stop streak")
d = decide_exit(pos(stop_streak=1), close=101.0, chand_atr=2.0, ema21=99.0,
                option_gain_pct=0.05, dte_remaining=75)
check("no breach -> HOLD, streak reset to 0", d.action == exits.HOLD and d.new_stop_streak == 0)

print("EXITS — Chandelier trail on the runner (armed)")
# peak 120, atr 3, mult 3 -> line 111 (breakeven floor 100 doesn't bind)
d1 = decide_exit(pos(trail_armed=True, peak_close=120.0, trail_streak=0),
                 close=110.0, chand_atr=3.0, ema21=118.0, option_gain_pct=0.5, dte_remaining=75)
check("below trail -> WAIT_TRAIL streak 1", d1.action == exits.WAIT_TRAIL and d1.new_trail_streak == 1)
check("trail line computed ~111", abs(d1.trail_line - 111.0) < 1e-6)
d2 = decide_exit(pos(trail_armed=True, peak_close=120.0, trail_streak=1),
                 close=110.0, chand_atr=3.0, ema21=118.0, option_gain_pct=0.5, dte_remaining=75)
check("2nd read -> TRAIL_EXIT", d2.action == exits.TRAIL_EXIT)
d3 = decide_exit(pos(trail_armed=True, peak_close=120.0, trail_streak=1),
                 close=116.0, chand_atr=3.0, ema21=118.0, option_gain_pct=0.6, dte_remaining=75)
check("above trail -> HOLD, trail streak reset", d3.action == exits.HOLD and d3.new_trail_streak == 0)

print("EXITS — PUT mirror (stop is ABOVE entry)")
dp = decide_exit(pos(option_type="PUT", entry_underlying=100.0, hard_stop_price=104.0,
                     peak_close=100.0, stop_streak=1),
                 close=105.0, chand_atr=2.0, ema21=101.0, option_gain_pct=-0.05, dte_remaining=75)
check("PUT close above stop -> HARD_STOP", dp.action == exits.HARD_STOP)

# ── Config clone_with (parameter sweeps) ─────────────────────────────
print("CONFIG — clone_with overrides without touching the global")
c = config.clone_with({"filters.adx_min": 20.0, "iv.vol_rank_max": 60.0})
check("clone has overridden adx_min", c.filters.adx_min == 20.0)
check("clone has overridden vol_rank_max", c.iv.vol_rank_max == 60.0)
check("GLOBAL config untouched (adx still 25)", config.filters.adx_min == 25.0)
try:
    config.clone_with({"filters.does_not_exist": 1}); bad = False
except KeyError:
    bad = True
check("unknown path raises KeyError", bad)

print("CONFIG — injected cfg changes sizing + delta target")
base_n = strategy.size_position(25000, 25000, 2000, 400).contracts           # risk 4% -> 2
rich = config.clone_with({"account.risk_per_trade_pct": 0.08})
rich_n = strategy.size_position(25000, 25000, 2000, 400, cfg=rich).contracts  # risk 8% -> 3 (capital-capped)
check("base risk sizes 2 contracts", base_n == 2)
check("doubled risk sizes more (3)", rich_n == 3)
deep = config.clone_with({"options.deep_itm_target_delta": 0.88})
check("injected cfg drives target_delta_for", strategy.target_delta_for(70, cfg=deep) == 0.88)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
