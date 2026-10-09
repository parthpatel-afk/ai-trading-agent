"""
The pretend portfolio for a backtest run.

Holds cash + open positions (one per symbol, like the live bot), applies buys / partial
scale-outs / full closes at MODELED option prices, and records ONE ledger row per fully-closed
trade (total P&L = banked partials + final proceeds − remaining cost) — the same accounting the
live `_book_close` uses. P&L is judged only on realized closes; open positions are marked just
for the equity curve.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List

from agent.exits import PositionState
from backtest.pricing import synth
from backtest.pricing.synth import SynthContract


@dataclass
class SimPosition:
    """An open option position + the exit state machine (mirrors the live DB Trade fields
    the monitor uses)."""
    symbol: str
    contract: SynthContract
    option_type: str                 # "CALL" | "PUT"
    contracts: int
    contracts_initial: int
    entry_price: float               # per share, as filled (incl. slippage)
    premium_paid: float              # total $ paid (incl. slippage)
    entry_date: date
    entry_underlying: float
    atr_at_entry: float
    hard_stop_price: float
    delta_at_entry: float
    hv_at_entry: float
    vol_rank_at_entry: Optional[float] = None
    adx_at_entry: Optional[float] = None
    # exit state
    peak_close: float = 0.0
    trail_armed: bool = False
    partial_done: bool = False
    stop_streak: int = 0
    trail_streak: int = 0
    realized_partial_pnl: float = 0.0

    def state(self) -> PositionState:
        return PositionState(self.option_type, self.entry_underlying, self.hard_stop_price,
                             self.peak_close, self.trail_armed, self.stop_streak, self.trail_streak)

    def mark(self, spot: float, asof: date) -> float:
        """Modeled per-share option value today."""
        return synth.reprice(self.contract, spot, asof)


class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: Dict[str, SimPosition] = {}
        self.ledger: List[dict] = []

    def position_for(self, symbol: str) -> Optional[SimPosition]:
        return self.positions.get(symbol)

    def equity(self, price_map: Dict[str, float]) -> float:
        """Cash + marked value of open positions. price_map: symbol → per-share option price."""
        inv = sum(p.contracts * price_map.get(p.symbol, p.entry_price) * 100
                  for p in self.positions.values())
        return self.cash + inv

    # ── mutations ────────────────────────────────────────────────────
    def open(self, pos: SimPosition) -> None:
        self.cash -= pos.premium_paid
        self.positions[pos.symbol] = pos

    def take_partial(self, pos: SimPosition, qty: int, opt_px: float) -> float:
        """Sell `qty` contracts at opt_px; bank the realized P&L on them. Returns banked $."""
        proceeds = opt_px * 100 * qty
        cost = pos.entry_price * 100 * qty
        banked = proceeds - cost
        pos.realized_partial_pnl += banked
        pos.contracts -= qty
        self.cash += proceeds
        return banked

    def close_full(self, pos: SimPosition, opt_px: float, exit_date: date,
                   exit_underlying: float, reason: str) -> dict:
        """Close the remaining contracts, book the trade, and append a ledger row."""
        proceeds = opt_px * 100 * pos.contracts
        cost_remaining = pos.entry_price * 100 * pos.contracts
        total = pos.realized_partial_pnl + (proceeds - cost_remaining)
        total_pct = total / pos.premium_paid if pos.premium_paid else 0.0
        self.cash += proceeds
        row = {
            "symbol": pos.symbol, "direction": pos.option_type,
            "entry_date": pos.entry_date, "exit_date": exit_date,
            "hold_days": (exit_date - pos.entry_date).days,
            "dte_at_entry": pos.contract.dte_at_entry,
            "strike": pos.contract.strike, "expiration": pos.contract.expiration,
            "entry_underlying": round(pos.entry_underlying, 2),
            "exit_underlying": round(exit_underlying, 2),
            "underlying_move_pct": round((exit_underlying / pos.entry_underlying - 1) *
                                         (1 if pos.option_type == "CALL" else -1), 4),
            "entry_opt_price": round(pos.entry_price, 4),
            "exit_opt_price": round(opt_px, 4),
            "contracts_initial": pos.contracts_initial,
            "delta_at_entry": pos.delta_at_entry,
            "vol_rank_at_entry": pos.vol_rank_at_entry,
            "adx_at_entry": pos.adx_at_entry,
            "hv_at_entry": round(pos.hv_at_entry, 4),
            "realized_partial_pnl": round(pos.realized_partial_pnl, 2),
            "pnl": round(total, 2), "pnl_pct": round(total_pct, 4),
            "exit_reason": reason,
        }
        self.ledger.append(row)
        self.positions.pop(pos.symbol, None)
        return row
