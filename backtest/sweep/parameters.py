"""
The sweepable parameters — friendly name → (config path, sensible default value range).

Each is a single dial the backtester can vary while holding everything else fixed (via
config.clone_with). This registry is also what a future product UI would expose as the
"tune this" controls. You can also sweep any dotted config path directly with custom values.
"""
from typing import Tuple, List

SWEEPABLE = {
    # name              (dotted config path,            default values to sweep)
    "adx_min":          ("filters.adx_min",             [15, 18, 20, 22, 25, 28, 32]),
    "vol_rank_max":     ("iv.vol_rank_max",             [40, 55, 70, 85, 100]),
    "target_delta":     ("options.target_delta",        [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]),
    "hard_stop_atr":    ("exits.hard_stop_atr_mult",    [1.0, 1.5, 2.0, 2.5, 3.0]),
    "chandelier_atr":   ("exits.chandelier_atr_mult",   [2.0, 2.5, 3.0, 3.5, 4.0]),
    "partial_profit":   ("exits.partial_profit_pct",    [0.50, 0.75, 1.00, 1.50]),
    "risk_per_trade":   ("account.risk_per_trade_pct",  [0.01, 0.02, 0.03, 0.04, 0.05]),
    "anti_chase_atr":   ("filters.chase_atr_max",       [0.75, 1.0, 1.25, 1.5, 2.0, 5.0]),
    "target_dte":       ("options.target_dte",          [45, 60, 75, 90, 120]),
    "time_exit_dte":    ("exits.time_exit_dte",         [7, 14, 21, 30]),
}


def resolve(param: str) -> Tuple[str, List]:
    """Return (dotted_path, default_values) for a registered name, or treat `param` as a raw
    dotted path (caller must then supply values)."""
    if param in SWEEPABLE:
        return SWEEPABLE[param]
    if "." in param:
        return param, []           # raw path; values must be provided explicitly
    raise KeyError(f"unknown parameter '{param}'. Known: {', '.join(SWEEPABLE)} (or a dotted path)")
