"""
Unit tests for the backtest synthetic option pricer (backtest/pricing/synth.py). No network.
Run: python scripts/test_pricing_synth.py
"""
import sys, os
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from agent import pricing
from backtest.pricing import synth

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

# ── HV ───────────────────────────────────────────────────────────────
print("HV — annualized historical volatility")
flat = pd.Series([100.0] * 60)
check("flat series -> ~0 vol", synth.hv_sigma(flat) is None or synth.hv_sigma(flat) < 1e-9)
rng = np.random.default_rng(0)
calm = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 300)))
wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.03, 300)))
hv_calm, hv_wild = synth.hv_sigma(calm), synth.hv_sigma(wild)
check("calm HV positive & sane", 0 < hv_calm < 0.30)
check("wild HV > calm HV", hv_wild > hv_calm)

# ── select_contract ──────────────────────────────────────────────────
print("SELECT — deep-ITM contract synthesis")
asof = date(2023, 1, 17)
c = synth.select_contract("TEST", "CALL", spot=100.0, sigma=0.30, asof=asof, target_delta=0.75)
check("contract created", c is not None)
check("realized delta ~ target 0.75", abs(c.entry_delta - 0.75) < 0.05)
check("CALL strike is ITM (< spot)", c.strike < 100.0)
check("DTE within band 45-120", 45 <= c.dte_at_entry <= 120)
check("expiration is a Friday", c.expiration.weekday() == 4)
check("entry price >= intrinsic", c.entry_price >= max(100.0 - c.strike, 0))
# Deeper ITM (higher delta) => smaller extrinsic FRACTION (the core reason deep-ITM is modelable).
def extrinsic_frac(ct, spot):
    return (ct.entry_price - max(spot - ct.strike, 0)) / ct.entry_price
c85 = synth.select_contract("TEST", "CALL", spot=100.0, sigma=0.30, asof=asof, target_delta=0.85)
check("deeper delta -> smaller extrinsic fraction",
      extrinsic_frac(c85, 100.0) < extrinsic_frac(c, 100.0))

cp = synth.select_contract("TEST", "PUT", spot=100.0, sigma=0.30, asof=asof, target_delta=0.75)
check("PUT strike is ITM (> spot)", cp.strike > 100.0)
check("PUT realized delta ~ 0.75", abs(cp.entry_delta - 0.75) < 0.05)

# ── reprice ──────────────────────────────────────────────────────────
print("REPRICE — path behavior")
at_entry = synth.reprice(c, 100.0, asof)
check("reprice at entry ~ entry price", abs(at_entry - c.entry_price) < 1e-3)  # entry_price stored @4dp
up = synth.reprice(c, 108.0, asof)
down = synth.reprice(c, 92.0, asof)
check("CALL rises when underlying rises", up > at_entry > down)
check("CALL move ~ delta*underlying move",
      abs((up - at_entry) - c.entry_delta * 8.0) < 3.0)   # roughly, ignoring gamma/theta
at_exp = synth.reprice(c, 130.0, c.expiration)
check("at expiration -> pure intrinsic", abs(at_exp - (130.0 - c.strike)) < 1e-6)
after_exp = synth.reprice(c, 130.0, date(2099, 1, 1))
check("after expiration -> intrinsic (dte floored at 0)", abs(after_exp - (130.0 - c.strike)) < 1e-6)

# PUT rises when underlying falls
pu = synth.reprice(cp, 92.0, asof)
pd_ = synth.reprice(cp, 108.0, asof)
check("PUT rises when underlying falls", pu > synth.reprice(cp, 100.0, asof) > pd_)

# consistency with the shared core
check("reprice == agent.pricing.bs_price",
      abs(synth.reprice(c, 103.0, asof) -
          pricing.bs_price(103.0, c.strike, (c.expiration - asof).days, c.entry_sigma, "CALL",
                           0.04)) < 1e-9)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
