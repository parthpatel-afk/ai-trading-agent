"""The backtest simulation engine — pretend portfolio + the day-by-day replay loop."""
from backtest.engine.simulator import run_backtest, BacktestResult
from backtest.engine.portfolio import Portfolio, SimPosition

__all__ = ["run_backtest", "BacktestResult", "Portfolio", "SimPosition"]
