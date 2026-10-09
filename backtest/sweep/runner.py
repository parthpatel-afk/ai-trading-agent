"""
Run a backtest across a range of values for ONE parameter (holding everything else fixed).

The output — a table (metrics per value) + a combined ledger tagged with `param_value` — is the
raw material for the 3D landscape (analysis/landscape.py): every trade knows both the parameter
value it was produced under and its entry date, so we can pivot into param × date-window × outcome.
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from agent.config import config
from backtest.engine import run_backtest
from backtest.sweep.parameters import resolve

_TABLE_METRICS = ["n_trades", "win_rate", "total_pnl", "return_pct", "avg_win", "avg_loss",
                  "profit_factor", "avg_hold_days", "calls", "puts", "max_drawdown_pct"]


@dataclass
class SweepResult:
    param: str
    path: str
    values: list
    table: pd.DataFrame        # one row per value: the summary metrics
    ledger: pd.DataFrame       # every trade across all values, with a 'param_value' column

    def best(self, metric: str = "total_pnl"):
        """The value that maximizes `metric` (for a quick read; robustness needs the full curve)."""
        if self.table.empty:
            return None
        row = self.table.loc[self.table[metric].idxmax()]
        return row["param_value"], row[metric]


def sweep_1d(param: str, values: Optional[List] = None, start: str = "2020-01-01",
             end: str = "2026-12-31", symbols: Optional[List[str]] = None,
             base_overrides: Optional[dict] = None, slippage_pct: float = 0.01,
             capital: Optional[float] = None, verbose: bool = True) -> SweepResult:
    path, default_values = resolve(param)
    values = values if values is not None else default_values
    if not values:
        raise ValueError(f"no values to sweep for '{param}' — pass values=[...]")

    rows, ledgers = [], []
    for v in values:
        overrides = {path: v, **(base_overrides or {})}
        cfg = config.clone_with(overrides)
        r = run_backtest(start=start, end=end, symbols=symbols, cfg=cfg,
                         slippage_pct=slippage_pct, initial_capital=capital)
        row = {"param_value": v, **{m: r.summary.get(m) for m in _TABLE_METRICS}}
        rows.append(row)
        if not r.ledger.empty:
            led = r.ledger.copy()
            led.insert(0, "param_value", v)
            ledgers.append(led)
        if verbose:
            print(f"  {param}={v:<6} -> {r.summary['n_trades']:>3} trades | "
                  f"P&L ${r.summary['total_pnl']:>+9,.0f} ({r.summary['return_pct']:>+.1%}) | "
                  f"win {r.summary['win_rate']:.0%}")

    table = pd.DataFrame(rows)
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return SweepResult(param, path, list(values), table, ledger)
