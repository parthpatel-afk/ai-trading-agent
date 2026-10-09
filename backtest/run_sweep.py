"""
Run a 1-parameter sweep, build the landscape, and render the interactive 3D surface + robustness
curve — the end-to-end "tune a dial and see the terrain" tool.

    python backtest/run_sweep.py --param adx_min
    python backtest/run_sweep.py --param vol_rank_max --metric total_pnl --period Q
    python backtest/run_sweep.py --param target_delta --values 0.6 0.7 0.8 --universe
    python backtest/run_sweep.py --param exits.hard_stop_atr_mult --values 1 1.5 2 2.5 3

Writes to backtest/results/sweep_<tag>/: table.csv, cube.csv, landscape_3d.html, robustness.html.
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from backtest import universe
from backtest.sweep import sweep_1d
from backtest.analysis import build_landscape
from backtest.viz import surface_3d, robustness_2d
from agent.paths import project_root

_PERIOD_LABEL = {"M": "month", "2M": "2-month", "Q": "quarter", "6M": "6-month", "Y": "year"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="adx_min", help="registered name or dotted config path")
    ap.add_argument("--values", nargs="*", type=float, default=None)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--metric", default="total_pnl")
    ap.add_argument("--period", default="Q", choices=list(_PERIOD_LABEL))
    ap.add_argument("--slippage", type=float, default=0.01)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    symbols = universe.symbols() if args.universe else args.symbols
    tag = args.tag or args.param.replace(".", "_")
    out = project_root() / "backtest" / "results" / f"sweep_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Sweeping {args.param} | {args.start}->{args.end} | "
          f"{'all 53' if args.universe else (len(symbols) if symbols else 18)} names\n")
    sr = sweep_1d(args.param, values=args.values, start=args.start, end=args.end,
                  symbols=symbols, slippage_pct=args.slippage)

    print("\nAll-time table:")
    print(sr.table.to_string(index=False))
    best_v, best_m = sr.best(args.metric)
    print(f"\nBest {args.metric}: {args.param}={best_v} ({best_m:,.0f}) "
          f"— check the landscape for whether it's a stable ridge or a lone spike.")

    cube = build_landscape(sr.ledger, metric=args.metric, period=args.period)
    sr.table.to_csv(out / "table.csv", index=False)
    cube.to_csv(out / "cube.csv", index=False)
    plabel = args.param
    mlabel = args.metric
    html3d = robust = None
    if not cube.empty:
        html3d = surface_3d(cube, plabel, mlabel, str(out / "landscape_3d.html"),
                            title=f"{mlabel} landscape — {plabel} × {_PERIOD_LABEL[args.period]} "
                                  f"({args.start[:4]}–{args.end[:4]})")
    robust = robustness_2d(sr.table, plabel, str(out / "robustness.html"), metric=args.metric)
    (out / "meta.json").write_text(json.dumps(
        {"param": args.param, "values": sr.values, "metric": args.metric, "period": args.period,
         "start": args.start, "end": args.end, "generated_at": datetime.now().isoformat(timespec="seconds")},
        indent=2, default=str))

    print(f"\nSaved -> {out}")
    if html3d:
        print(f"  3D landscape : {html3d}")
    print(f"  robustness   : {robust}")


if __name__ == "__main__":
    main()
