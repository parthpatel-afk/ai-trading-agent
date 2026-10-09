"""
Build the landscape cube from a sweep's ledger — the core idea:

    X = the parameter value        (e.g. the ADX gate, or the IV/vol cap)
    Z = a date-window bucket       (trades grouped by ENTRY month/quarter, tunable)
    Y = an outcome for that cell   (total P&L / avg P&L / # trades / win-rate)

Trades are bucketed by their ENTRY date (the parameter is a BUY-time decision),
regardless of when they closed. The cube is "long form" (one row per param×period cell); `to_grid`
pivots it into the matrix a 3D surface wants. Missing param×period cells = 0 (no trades → no P&L).
"""
from typing import Callable, Dict
import pandas as pd

# metric name -> how to aggregate a group of trades' pnl into one cell value
METRICS: Dict[str, Callable[[pd.DataFrame], float]] = {
    "total_pnl":  lambda g: float(g["pnl"].sum()),
    "avg_pnl":    lambda g: float(g["pnl"].mean()),
    "n_trades":   lambda g: int(len(g)),
    "win_rate":   lambda g: float((g["pnl"] > 0).mean()),
    "return_pct": lambda g: float(g["pnl"].sum()),   # scaled to % by caller if desired
}


def build_landscape(ledger: pd.DataFrame, metric: str = "total_pnl",
                    period: str = "Q", param_col: str = "param_value") -> pd.DataFrame:
    """Long-form cube: columns [param_value, period, value]. `period` is a pandas offset alias
    ('M' month, 'Q' quarter, '2M' two-month, '6M', 'Y'). Empty ledger → empty frame."""
    if ledger is None or ledger.empty:
        return pd.DataFrame(columns=[param_col, "period", "value"])
    if metric not in METRICS:
        raise KeyError(f"unknown metric '{metric}'. Known: {', '.join(METRICS)}")
    df = ledger.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["period"] = df["entry_date"].dt.to_period(period).dt.to_timestamp()
    fn = METRICS[metric]
    rows = [{param_col: pv, "period": per, "value": fn(g)}
            for (pv, per), g in df.groupby([param_col, "period"])]
    return pd.DataFrame(rows).sort_values([param_col, "period"]).reset_index(drop=True)


def to_grid(cube: pd.DataFrame, param_col: str = "param_value",
            fill: float = 0.0) -> pd.DataFrame:
    """Pivot the long cube into a matrix: index = period (Z), columns = param value (X),
    cells = metric (Y/height). Missing param×period cells filled with `fill` (usually 0)."""
    if cube.empty:
        return pd.DataFrame()
    grid = cube.pivot(index="period", columns=param_col, values="value")
    # ensure every period row exists across the full monthly/quarterly span (no gaps in Z)
    return grid.sort_index().fillna(fill)
