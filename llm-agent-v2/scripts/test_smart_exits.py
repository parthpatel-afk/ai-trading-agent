"""
Simulation test for check_option_exit() and check_new_entry().
Run from project root: python scripts/test_smart_exits.py
"""
import sys
import io
import os

# Fix Windows console encoding so special chars don't blow up
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Minimal config stubs (no DB / env needed) ─────────────────────────────────
from unittest.mock import patch, MagicMock

options_cfg = MagicMock()
options_cfg.time_exit_dte = 21       # updated: expert 21 DTE close rule
options_cfg.stop_loss_pct = 0.50
options_cfg.trailing_stop_pct = 0.20

risk_cfg = MagicMock()
risk_cfg.max_positions = 6
risk_cfg.max_position_pct = 0.08
risk_cfg.cash_reserve_pct = 0.20
risk_cfg.daily_loss_limit_pct = 0.08
risk_cfg.portfolio_drawdown_limit_pct = 0.15
risk_cfg.min_premium = 75
risk_cfg.max_premium = 800
risk_cfg.initial_capital = 10000

mock_config = MagicMock()
mock_config.options = options_cfg
mock_config.risk = risk_cfg

with patch.dict("sys.modules", {"agent.config": MagicMock(config=mock_config)}):
    import agent.config as _ac
    _ac.config = mock_config

    from agent.risk import RiskEngine, _bs_delta, _theta_pct_per_day

# ── Test runner ───────────────────────────────────────────────────────────────

results = []

def check(label, got, expect_exit, expect_reason_contains):
    ok = (got.should_exit == expect_exit) and (expect_reason_contains in got.reason)
    results.append(("PASS" if ok else "FAIL", label))
    marker = "OK" if ok else "!!"
    print(f"  [{marker}] {label}")
    if not ok:
        print(f"       expected exit={expect_exit}, reason contains '{expect_reason_contains}'")
        print(f"       got    exit={got.should_exit}, reason='{got.reason}'")

def check_val(label, got, expected, tol=0.005):
    ok = abs(got - expected) <= tol
    results.append(("PASS" if ok else "FAIL", label))
    marker = "OK" if ok else "!!"
    print(f"  [{marker}] {label}: got {got:.4f}, expected ~{expected:.4f}")

def check_risk(label, got, expect_approved, expect_reason_contains=""):
    ok = got.approved == expect_approved
    if expect_reason_contains:
        ok = ok and (expect_reason_contains in got.reason)
    results.append(("PASS" if ok else "FAIL", label))
    marker = "OK" if ok else "!!"
    print(f"  [{marker}] {label}")
    if not ok:
        print(f"       expected approved={expect_approved}, reason contains '{expect_reason_contains}'")
        print(f"       got    approved={got.approved}, reason='{got.reason}'")


risk = RiskEngine()

# ── Section 1: Helper function correctness ────────────────────────────────────
print("\n=== 1. Helper Functions ===")

check_val("theta @ 45 DTE",  _theta_pct_per_day(45), 0.033)
check_val("theta @ 30 DTE",  _theta_pct_per_day(30), 0.067)
check_val("theta @ 21 DTE",  _theta_pct_per_day(21), 0.111)
check_val("theta @ 14 DTE",  _theta_pct_per_day(14), 0.150)
check_val("theta @  7 DTE",  _theta_pct_per_day(7),  0.250)
check_val("theta @  3 DTE",  _theta_pct_per_day(3),  0.250)

deep_itm_delta = _bs_delta(S=200, K=100, T=30/365, r=0.045, sigma=0.30, option_type="call")
check_val("delta deep ITM call (S=200 K=100)", deep_itm_delta, 1.0, tol=0.05)

deep_otm_delta = _bs_delta(S=100, K=200, T=30/365, r=0.045, sigma=0.30, option_type="call")
check_val("delta deep OTM call (S=100 K=200)", deep_otm_delta, 0.0, tol=0.05)

atm_put_delta = _bs_delta(S=150, K=150, T=30/365, r=0.045, sigma=0.30, option_type="put")
check_val("delta ATM put (S=K)", atm_put_delta, -0.50, tol=0.06)

assert _bs_delta(0, 150, 30/365, 0.045, 0.30) == 0.0, "zero S should return 0.0"
assert _bs_delta(150, 150, 0, 0.045, 0.30) == 0.0,    "zero T should return 0.0"
assert _bs_delta(150, 150, 30/365, 0.045, 0) == 0.0,  "zero sigma should return 0.0"
print("  [OK] bad inputs -> 0.0")


# ── Section 2: Time exit and stop loss ───────────────────────────────────────
print("\n=== 2. Time Exit (21 DTE) and Stop Loss ===")

# At 20 DTE (below threshold of 21): close
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=800,
                              dte_remaining=20, peak_value=1000)
check("time_exit at 20 DTE", sig, True, "time_exit")

# At exactly 21 DTE: close
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=900,
                              dte_remaining=21, peak_value=1000)
check("time_exit at 21 DTE (at limit)", sig, True, "time_exit")

# At 22 DTE: should NOT close on time exit alone
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=900,
                              dte_remaining=22, peak_value=1000)
check("no time_exit at 22 DTE (above limit)", sig, False, "within parameters")

# Stop loss at exactly 50%
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=500,
                              dte_remaining=30, peak_value=1000)
check("stop_loss at 50% loss", sig, True, "stop_loss")

# Stop loss above 50%
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=400,
                              dte_remaining=30, peak_value=1000)
check("stop_loss at 60% loss", sig, True, "stop_loss")

# Trailing stop: peak +250%, pulled back 43% from peak (threshold 25% at this gain tier)
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=2425,
                              dte_remaining=30, peak_value=3500)
check("trailing_stop peak +250% pulled back 43%", sig, True, "trailing_stop")

# No exit: small loss, plenty of DTE
sig = risk.check_option_exit("TEST", premium_paid=1000, current_value=900,
                              dte_remaining=30, peak_value=1000)
check("no exit: small loss, plenty of DTE", sig, False, "within parameters")


# ── Section 3: Delta Floor ────────────────────────────────────────────────────
print("\n=== 3. Delta Floor Check (< 0.25) ===")

# Far below strike (call): delta near 0 -> exit
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=600, dte_remaining=30, peak_value=1000,
    stock_price=100, strike=160, iv=0.30, option_type="call"
)
check("delta_floor far OTM call (S=100 K=160)", sig, True, "delta_floor")

# Far above strike (put): delta near 0 -> exit
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=600, dte_remaining=30, peak_value=1000,
    stock_price=200, strike=120, iv=0.30, option_type="put"
)
check("delta_floor far OTM put (S=200 K=120)", sig, True, "delta_floor")

# ATM call: delta ~0.50, should NOT trigger
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=600, dte_remaining=30, peak_value=1000,
    stock_price=150, strike=150, iv=0.30, option_type="call"
)
check("no delta_floor: ATM call (delta ~0.50)", sig, False, "within parameters")

# Stop loss fires BEFORE delta floor
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=400, dte_remaining=30, peak_value=1000,
    stock_price=100, strike=160, iv=0.30, option_type="call"
)
check("stop_loss beats delta_floor (priority)", sig, True, "stop_loss")

# Time exit fires BEFORE delta floor
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=600, dte_remaining=20, peak_value=1000,
    stock_price=100, strike=160, iv=0.30, option_type="call"
)
check("time_exit beats delta_floor (priority)", sig, True, "time_exit")

# No stock price: delta floor skipped
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=600, dte_remaining=30, peak_value=1000
)
check("delta_floor skipped when no stock_price", sig, False, "within parameters")


# ── Section 4: Velocity Deterioration ────────────────────────────────────────
print("\n=== 4. Velocity Deterioration Check ===")

# Losing much faster than theta explains at 45 DTE
# avg_daily_loss=(1000-510)/10=49/day, theta=510*0.033=16.8/day, threshold=25.3/day -> EXIT
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=510, dte_remaining=45, peak_value=1000,
    stock_price=150, strike=160, iv=0.30, option_type="call", days_held=10
)
check("velocity_exit: losing 49/day vs theta 25/day at 45 DTE", sig, True, "velocity_exit")

# Only 2 days held: should NOT fire (need >= 3)
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=510, dte_remaining=45, peak_value=1000,
    stock_price=150, strike=160, iv=0.30, option_type="call", days_held=2
)
check("no velocity_exit: only 2 days held (need >= 3)", sig, False, "within parameters")

# Position profitable: should NOT fire
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=1200, dte_remaining=45, peak_value=1200,
    stock_price=165, strike=160, iv=0.30, option_type="call", days_held=10
)
check("no velocity_exit: position is profitable", sig, False, "within parameters")

# Mild loss within theta: should NOT fire
sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=900, dte_remaining=45, peak_value=1000,
    stock_price=155, strike=160, iv=0.30, option_type="call", days_held=10
)
check("no velocity_exit: mild loss within theta range", sig, False, "within parameters")


# ── Section 5: No-param fallback ─────────────────────────────────────────────
print("\n=== 5. Fallback: No Smart Params Provided ===")

sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=800, dte_remaining=30, peak_value=1000
)
check("no smart params, normal hold", sig, False, "within parameters")

sig = risk.check_option_exit(
    "TEST", premium_paid=1000, current_value=490, dte_remaining=30, peak_value=1000
)
check("stop_loss fires with no smart params", sig, True, "stop_loss")


# ── Section 6: Position sizing — reject oversized trades ─────────────────────
print("\n=== 6. Position Sizing — Reject if 1 Contract Exceeds Max ===")

# equity=$10000, max_position_pct=8% -> max=$800
# TSLA at $2375/contract: should be REJECTED
result = risk.check_new_entry(
    symbol="TSLA",
    current_equity=10000,
    available_cash=8000,
    num_open_positions=0,
    estimated_premium=2375,
)
check_risk("TSLA $2375/contract rejected (exceeds $800 max)", result, False, "exceeds max position")

# SMCI at $395/contract: should be APPROVED (within $800)
result = risk.check_new_entry(
    symbol="SMCI",
    current_equity=10000,
    available_cash=8000,
    num_open_positions=0,
    estimated_premium=395,
)
check_risk("SMCI $395/contract approved (within $800 max)", result, True)

# XLY at $969/contract: should be REJECTED (over $800)
result = risk.check_new_entry(
    symbol="XLY",
    current_equity=10000,
    available_cash=8000,
    num_open_positions=0,
    estimated_premium=969,
)
check_risk("XLY $969/contract rejected (exceeds $800 max)", result, False, "exceeds max position")

# At max positions: should be REJECTED regardless of premium
result = risk.check_new_entry(
    symbol="TEST",
    current_equity=10000,
    available_cash=8000,
    num_open_positions=6,
    estimated_premium=200,
)
check_risk("rejected at max positions (6/6)", result, False, "Max positions")

# Affordable contract but insufficient cash after reserve
# cash_reserve = 20% of $10k = $2000, tradeable = $500 - $2000 = negative -> reject
result = risk.check_new_entry(
    symbol="TEST",
    current_equity=10000,
    available_cash=1000,  # only $1000 cash, reserve eats $2000
    num_open_positions=0,
    estimated_premium=200,
)
check_risk("rejected: insufficient cash after reserve", result, False)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
total = len(results)
passed = sum(1 for s, _ in results if s == "PASS")
failed = total - passed
print(f"=== Results: {passed}/{total} passed ===")
if failed:
    print(f"\nFailed tests:")
    for s, label in results:
        if s != "PASS":
            print(f"  - {label}")
    sys.exit(1)
else:
    print("All tests passed.")
