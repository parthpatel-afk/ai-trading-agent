"""
Tests for the backtest engine (portfolio accounting + simulator faithfulness).
Uses the cached data for the faithfulness checks. Run: python scripts/test_engine.py
"""
import sys, os
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger; logger.remove()  # quiet warmup warnings
from backtest.engine import run_backtest
from backtest.engine.portfolio import Portfolio, SimPosition
from backtest.pricing.synth import SynthContract

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")


# ── Portfolio accounting ─────────────────────────────────────────────
print("PORTFOLIO — cash / partial / close arithmetic")
c = SynthContract("X", "CALL", strike=90.0, expiration=date(2025, 3, 21), dte_at_entry=75,
                  entry_sigma=0.30, entry_price=12.0, entry_spot=100.0, entry_delta=0.75)
pos = SimPosition(symbol="X", contract=c, option_type="CALL", contracts=3, contracts_initial=3,
                  entry_price=12.0, premium_paid=12.0 * 100 * 3, entry_date=date(2025, 1, 6),
                  entry_underlying=100.0, atr_at_entry=2.0, hard_stop_price=96.0,
                  delta_at_entry=0.75, hv_at_entry=0.30, peak_close=100.0)
port = Portfolio(25000.0)
port.open(pos)
check("open() deducts premium from cash", abs(port.cash - (25000 - 3600)) < 1e-6)
banked = port.take_partial(pos, 1, opt_px=20.0)     # sell 1 @ 20 (cost 12) -> +800
check("partial banks (20-12)*100 = 800", abs(banked - 800) < 1e-6)
check("partial adds proceeds to cash", abs(port.cash - (21400 + 2000)) < 1e-6)
check("partial reduces contracts to 2", pos.contracts == 2)
row = port.close_full(pos, opt_px=25.0, exit_date=date(2025, 2, 3),
                      exit_underlying=115.0, reason="chandelier_trail")
# total = banked 800 + (25-12)*100*2 = 800 + 2600 = 3400
check("close books total incl. partials (3400)", abs(row["pnl"] - 3400) < 1e-6)
check("cash back to 23400 + 5000 = 28400", abs(port.cash - 28400) < 1e-6)
check("position removed after full close", port.position_for("X") is None)
check("ledger has one row", len(port.ledger) == 1)

# ── Simulator faithfulness: reproduces the live 0-trade window ────────
print("SIMULATOR — trust test (reproduces live 06-29..07-10: 0 trades, ADX-blocked)")
r = run_backtest(start="2026-06-29", end="2026-07-10", record_decisions=True)
check("0 trades over the choppy live window", r.summary["n_trades"] == 0)
fresh = r.decisions[r.decisions["fresh"] == True]
check("did see fresh cross signals", len(fresh) > 5)
check("every fresh signal rejected (ADX etc.)", (~fresh["action"].str.startswith("BUY")).all())
# spot-check a known live reading: 07-06 SPY ADX ~22.2 (live log 07-07)
spy = fresh[(fresh["symbol"] == "SPY") & (fresh["date"].astype(str) == "2026-07-06")]
check("SPY 07-06 ADX reproduces live ~22.2", len(spy) == 1 and abs(spy.iloc[0]["adx"] - 22.2) < 0.3)

# ── Ledger internal consistency on a multi-trade run ─────────────────
print("SIMULATOR — ledger/equity consistency (2020->2026, default watchlist)")
rr = run_backtest(start="2020-01-01", end="2026-07-24")
s = rr.summary
check("final_equity == capital + total_pnl",
      abs(s["final_equity"] - (s["initial_capital"] + s["total_pnl"])) < 1.0)
check("ledger pnl sum == summary total_pnl",
      abs(float(rr.ledger["pnl"].sum()) - s["total_pnl"]) < 1.0)
check("equity curve ends ~ final_equity",
      abs(float(rr.equity_curve["equity"].iloc[-1]) - s["final_equity"]) < 5.0)
check("produced some trades (real trends exist in 2020-2026)", s["n_trades"] >= 1)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
