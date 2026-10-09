"""
Run a single backtest and (optionally) save its ledger / equity curve / summary.

    python backtest/run_backtest.py                                  # default 18 names, 2020->now
    python backtest/run_backtest.py --start 2023-01-01 --save run23
    python backtest/run_backtest.py --symbols AAPL MSFT NVDA --save tech
    python backtest/run_backtest.py --universe                        # all 53 names

Saves to backtest/results/<tag>/ (git-ignored). Phase 4 (sweeps) and Phase 5 (the 3D
landscape) build on the ledgers this produces.
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from backtest import universe
from backtest.engine import run_backtest
from agent.paths import project_root


def _results_dir(tag: str):
    d = project_root() / "backtest" / "results" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe", action="store_true", help="use all 53 names")
    ap.add_argument("--slippage", type=float, default=0.01)
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--save", default=None, help="tag; saves ledger/equity/summary under results/<tag>")
    args = ap.parse_args()

    symbols = universe.symbols() if args.universe else args.symbols
    print(f"Backtest {args.start} -> {args.end} | "
          f"{'all 53' if args.universe else (len(symbols) if symbols else 18)} names | "
          f"slippage {args.slippage:.1%}\n")

    r = run_backtest(start=args.start, end=args.end, symbols=symbols,
                     slippage_pct=args.slippage, initial_capital=args.capital,
                     record_decisions=bool(args.save))
    print(repr(r), "\n")
    for k in ("n_trades", "win_rate", "total_pnl", "return_pct", "final_equity",
              "avg_win", "avg_loss", "profit_factor", "avg_hold_days", "calls", "puts",
              "max_drawdown_pct"):
        print(f"  {k:18} {r.summary.get(k)}")
    print("  exit_reasons:", r.summary.get("exit_reasons"))

    if args.save:
        d = _results_dir(args.save)
        r.ledger.to_csv(d / "ledger.csv", index=False)
        r.equity_curve.to_csv(d / "equity.csv")
        if not r.decisions.empty:
            r.decisions.to_csv(d / "decisions.csv", index=False)
        (d / "summary.json").write_text(json.dumps(
            {**r.summary, "generated_at": datetime.now().isoformat(timespec="seconds"),
             "slippage_pct": args.slippage}, indent=2, default=str))
        print(f"\nSaved ledger + equity + summary -> {d}")


if __name__ == "__main__":
    main()
