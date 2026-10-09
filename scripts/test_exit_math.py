"""
Unit tests for Trend-Buyer v1 exit + sizing math (no network). Run: python scripts/test_exit_math.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import strategy
from agent.config import config

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  ✅ {name}")
    else:
        failed += 1; print(f"  ❌ {name}")

print("HARD STOP LEVEL (entry=100, ATR=2, mult=%.1f)" % config.exits.hard_stop_atr_mult)
m = config.exits.hard_stop_atr_mult
check("CALL stop below entry", strategy.hard_stop_level(100, 2, "CALL") == 100 - m*2)
check("PUT stop above entry",  strategy.hard_stop_level(100, 2, "PUT")  == 100 + m*2)

print("CHANDELIER (extreme=120, ATR=3, mult=%.1f)" % config.exits.chandelier_atr_mult)
k = config.exits.chandelier_atr_mult
check("CALL line below peak",   strategy.chandelier_level(120, 3, "CALL") == 120 - k*3)
check("PUT line above trough",  strategy.chandelier_level(80, 3, "PUT")   == 80 + k*3)

print("BREAKEVEN FLOOR (entry=100)")
check("CALL floor lifts low line to entry", strategy.apply_breakeven_floor(95, 100, "CALL") == 100)
check("CALL floor keeps high line",         strategy.apply_breakeven_floor(110, 100, "CALL") == 110)
check("PUT floor caps high line at entry",  strategy.apply_breakeven_floor(105, 100, "PUT") == 100)

print("BREACH CHECK")
check("CALL breaches at/below line", strategy.stop_or_trail_breached(94, 95, "CALL") is True)
check("CALL holds above line",       strategy.stop_or_trail_breached(96, 95, "CALL") is False)
check("PUT breaches at/above line",  strategy.stop_or_trail_breached(106, 105, "PUT") is True)
check("PUT holds below line",        strategy.stop_or_trail_breached(104, 105, "PUT") is False)

print("STOP-LOSS ESTIMATE (delta=0.75, ATR=5, mult=2)")
est = strategy.estimate_stop_loss_per_contract(0.75, 5, 2)
check("≈ 0.75*10*100 = 750", abs(est - 750) < 1e-6)

print("SIZING ($25k equity, $25k cash)")
eq = cash = 25000.0
# cheap-ish contract: $2000/ct, est stop $400 -> risk cap 0.04*25000=1000 -> by_risk=2; capital cap 0.25*25k=6250 -> 3; -> 2
s = strategy.size_position(eq, cash, 2000, 400)
check("normal sizes by risk (2)", s.contracts == 2)
# chunky deep-ITM: $4845/ct, est stop $1323 -> risk cap 1000 -> by_risk 0; allow 1 (within 6% ceiling 1500 & capital 6250 & cash) -> 1
s = strategy.size_position(eq, cash, 4845, 1323)
check("chunky allows 1 (within ceilings)", s.contracts == 1)
# too expensive: $6745/ct > capital cap 6250 -> reject 0
s = strategy.size_position(eq, cash, 6745, 1695)
check("over capital cap -> reject 0", s.contracts == 0)
# stop-loss over 6% ceiling on 1 ct: est 2000 > 1500 -> reject
s = strategy.size_position(eq, cash, 3000, 2000)
check("over risk ceiling -> reject 0", s.contracts == 0)

print("TARGET DELTA (deeper when vol expensive)")
check("cheap vol -> base delta", strategy.target_delta_for(20) == config.options.target_delta)
check("rich vol  -> deeper delta", strategy.target_delta_for(70) == config.options.deep_itm_target_delta)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
