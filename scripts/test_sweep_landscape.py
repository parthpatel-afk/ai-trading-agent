"""
Tests for the sweep runner + landscape cube (Phase 4/5). Run: python scripts/test_sweep_landscape.py
The landscape math is tested on a synthetic ledger (no backtest); one tiny real sweep checks wiring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger; logger.remove()
import pandas as pd
from backtest.analysis import build_landscape, to_grid
from backtest.sweep import sweep_1d, resolve

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

# ── Landscape math on a synthetic ledger ─────────────────────────────
print("LANDSCAPE — bucketing + aggregation")
led = pd.DataFrame([
    {"param_value": 20, "entry_date": "2024-01-15", "pnl": 100.0},
    {"param_value": 20, "entry_date": "2024-02-10", "pnl": -50.0},
    {"param_value": 20, "entry_date": "2024-04-05", "pnl": 200.0},
    {"param_value": 25, "entry_date": "2024-01-20", "pnl": 300.0},
])
cube = build_landscape(led, metric="total_pnl", period="Q")
def cell(c, pv, per):
    r = c[(c["param_value"] == pv) & (c["period"] == pd.Timestamp(per))]
    return float(r["value"].iloc[0]) if len(r) else None
check("param20 Q1'24 sums 100 + -50 = 50", cell(cube, 20, "2024-01-01") == 50.0)
check("param20 Q2'24 = 200", cell(cube, 20, "2024-04-01") == 200.0)
check("param25 Q1'24 = 300", cell(cube, 25, "2024-01-01") == 300.0)

n = build_landscape(led, metric="n_trades", period="Q")
check("n_trades param20 Q1 = 2", cell(n, 20, "2024-01-01") == 2)
wr = build_landscape(led, metric="win_rate", period="Q")
check("win_rate param20 Q1 = 0.5", cell(wr, 20, "2024-01-01") == 0.5)

print("LANDSCAPE — to_grid pivot")
grid = to_grid(cube)
check("grid shape = (2 periods, 2 params)", grid.shape == (2, 2))
check("missing param25 Q2 filled with 0", grid.loc[pd.Timestamp("2024-04-01"), 25] == 0.0)
check("empty ledger -> empty cube", build_landscape(pd.DataFrame(), "total_pnl").empty)

# ── Registry ─────────────────────────────────────────────────────────
print("SWEEP — parameter registry")
path, vals = resolve("adx_min")
check("adx_min resolves to filters.adx_min", path == "filters.adx_min" and 25 in vals)
check("raw dotted path passes through", resolve("exits.time_exit_dte")[0] == "exits.time_exit_dte")

# ── One tiny real sweep (wiring) ─────────────────────────────────────
print("SWEEP — tiny real run wiring (SPY+AAPL, 2025, adx 20 vs 25)")
sr = sweep_1d("adx_min", values=[20, 25], start="2025-01-01", end="2026-01-01",
              symbols=["SPY", "AAPL"], verbose=False)
check("table has a row per value", len(sr.table) == 2)
check("table carries metrics", "total_pnl" in sr.table.columns and "n_trades" in sr.table.columns)
check("ledger tagged with param_value",
      sr.ledger.empty or "param_value" in sr.ledger.columns)
check("best() returns a value present in the sweep", sr.best("total_pnl")[0] in [20, 25])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
