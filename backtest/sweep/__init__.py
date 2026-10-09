"""Parameter sweeps — run the backtest across many settings, collect ledgers for the landscape."""
from backtest.sweep.runner import sweep_1d, SweepResult
from backtest.sweep.parameters import SWEEPABLE, resolve

__all__ = ["sweep_1d", "SweepResult", "SWEEPABLE", "resolve"]
