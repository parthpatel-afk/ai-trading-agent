"""
Unit tests for the 2026-06-12 exit fixes — pure-logic checks that don't need a live market.

Covers:
  1) Real-fill cost-to-close math used by executor.close_credit_spread
  2) Honest P&L from fills vs the old mark-based number (the NVDA −$120-that-was-−$10)
  3) The stop-loss guard flow in _check_credit_spread_exit (warmup + N-reading confirm)

NOTE: the actual MLEG fill-reading and order submission can only be integration-tested on a
live paper close during market hours. These tests lock the math/decision logic.

Run:  python scripts/test_close_logic.py
"""


def cost_to_close_from_fills(legs, contracts):
    """Mirror of executor.close_credit_spread fill math.
    legs: list of (side, filled_avg_price). BUY (closing the short) costs us;
    SELL (closing the long) pays us back."""
    net = 0.0
    for side, fp in legs:
        net += fp if side.endswith("buy") else -fp
    return round(net * 100 * contracts, 2)


def stop_guard(is_stop, mins_since_open, streak, warmup=5, need=2):
    """Mirror of the _check_credit_spread_exit stop guard. Returns (action, new_streak)."""
    if not is_stop:
        return ("act", 0)
    if mins_since_open is not None and mins_since_open < warmup:
        return ("skip_warmup", streak)
    streak += 1
    if streak < need:
        return ("wait_confirm", streak)
    return ("fire", streak)


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")
    return ok


def main():
    passed = True

    print("1) Real-fill cost-to-close (NVDA: buy short 220C @4.80, sell long 225C @3.75, x1)")
    cost = cost_to_close_from_fills([("buy", 4.80), ("sell", 3.75)], 1)
    passed &= check("cost_to_close", cost, 105.0)

    print("2) Honest P&L vs old mark-based number (credit collected = $95)")
    credit = 95.0
    real_pnl = round(credit - cost, 2)            # off actual fills
    mark_pnl = round(credit - 215.0, 2)           # what the bot booked off the junk mark
    passed &= check("real P&L (fills)", real_pnl, -10.0)
    passed &= check("old P&L (mark)", mark_pnl, -120.0)
    print(f"      -> fills say -$10, the bot booked -$120. Fix records the honest -$10.")

    print("3) Multi-contract fill math (XLF-style x6: buy short @0.50, sell long @0.30)")
    passed &= check("cost x6", cost_to_close_from_fills([("buy", 0.50), ("sell", 0.30)], 6), 120.0)

    print("4) Stop guard — NVDA fired 4s after open (0.07 min): must be SKIPPED")
    action, _ = stop_guard(is_stop=True, mins_since_open=0.07, streak=0)
    passed &= check("opening-minute stop", action, "skip_warmup")

    print("5) Stop guard — after warmup needs 2 readings in a row")
    a1, s1 = stop_guard(is_stop=True, mins_since_open=30, streak=0)   # 1st reading
    a2, s2 = stop_guard(is_stop=True, mins_since_open=30, streak=s1)  # 2nd reading
    passed &= check("1st reading", a1, "wait_confirm")
    passed &= check("2nd reading", a2, "fire")

    print("6) Stop guard — a one-off bad tick that clears next cycle never fires")
    a1, s1 = stop_guard(is_stop=True, mins_since_open=30, streak=0)   # bad tick
    a2, _ = stop_guard(is_stop=False, mins_since_open=30, streak=s1)  # back under line -> reset
    passed &= check("bad tick then healthy", (a1, a2), ("wait_confirm", "act"))

    print("7) Non-stop exits (profit/time) act immediately, unaffected by guards")
    passed &= check("profit/time exit", stop_guard(is_stop=False, mins_since_open=0.07, streak=0)[0], "act")

    print()
    print("ALL PASSED" if passed else "SOME FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
