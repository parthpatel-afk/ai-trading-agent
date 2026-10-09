"""
AI Investor v2 — Database Layer
Slim SQLite schema for the EMA crossover strategy.

Three tables:
  - Trade            : every option trade (open + closed)
  - SignalLog        : per-symbol EMA evaluation + action taken (powers the dashboard)
  - PortfolioSnapshot: equity/cash/positions over time (equity curve)
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger

Base = declarative_base()


# ============================================================
# MODELS
# ============================================================

class Trade(Base):
    """Every option trade the bot makes — open and closed."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)    # underlying, e.g. "AAPL"
    side = Column(String(4), nullable=False, default="buy")

    contract_symbol = Column(String(25), nullable=True)        # OCC symbol
    option_type = Column(String(4), nullable=True)             # "CALL" | "PUT"
    strategy_type = Column(String(15), nullable=True)          # "BUY_CALL" | "BUY_PUT"
    strike = Column(Float, nullable=True)
    expiration = Column(String(10), nullable=True)             # "YYYY-MM-DD"
    dte_at_entry = Column(Integer, nullable=True)
    iv_at_entry = Column(Float, nullable=True)
    contracts = Column(Integer, default=1)

    entry_price = Column(Float, nullable=True)                 # premium per share at entry
    exit_price = Column(Float, nullable=True)                  # premium per share at exit
    premium_paid = Column(Float, nullable=True)                # total cost (entry_price × 100 × contracts)

    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)

    status = Column(String(10), default="open")                # "open" | "closed"
    pnl = Column(Float, nullable=True)                          # total realized (partials + final)
    pnl_pct = Column(Float, nullable=True)
    exit_reason = Column(String(60), nullable=True)            # "hard_stop" | "chandelier_trail" | "time_exit" | "ema_flip"

    # ── Trend-Buyer v1 exit state machine (all underlying-based) ──
    delta_at_entry = Column(Float, nullable=True)
    vol_rank_at_entry = Column(Float, nullable=True)
    entry_underlying = Column(Float, nullable=True)             # stock price at entry (stop reference)
    atr_at_entry = Column(Float, nullable=True)                 # ATR used for the hard stop
    contracts_initial = Column(Integer, nullable=True)         # original count (for partial tracking)
    hard_stop_price = Column(Float, nullable=True)             # underlying level for the initial stop
    partial_done = Column(Boolean, default=False)             # has the partial scale-out fired?
    trail_armed = Column(Boolean, default=False)              # is the Chandelier trail active?
    peak_close = Column(Float, nullable=True)                  # highest(call)/lowest(put) close since entry
    stop_streak = Column(Integer, default=0)                  # consecutive confirming closes (hard stop)
    trail_streak = Column(Integer, default=0)                 # consecutive closes beyond the trail line
    realized_partial_pnl = Column(Float, default=0.0)         # cash banked from partial sells

    alpaca_order_id = Column(String(50), nullable=True)
    strategy_version = Column(String(15), default="trendbuyer-v1")
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalLog(Base):
    """One row per symbol per daily evaluation — what the EMA said and what we did."""
    __tablename__ = "signal_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(10), nullable=False, index=True)

    close = Column(Float, nullable=True)
    ema_fast = Column(Float, nullable=True)
    ema_slow = Column(Float, nullable=True)
    fast_above_slow = Column(Boolean, nullable=True)
    cross = Column(String(4), nullable=True)                   # "BULL" | "BEAR" | None
    cross_age_days = Column(Integer, nullable=True)
    fresh = Column(Boolean, default=False)

    action = Column(String(12), nullable=True)                 # "BUY_CALL" | "BUY_PUT" | "HOLD" | "CLOSE" | "SKIP"
    reason = Column(Text, nullable=True)
    executed = Column(Boolean, default=False)
    metrics_json = Column(JSON, nullable=True)                  # filter metrics: adx, vol_rank, atr, regime, extension


class PortfolioSnapshot(Base):
    """Portfolio state over time — equity curve + position count for the dashboard."""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    invested_value = Column(Float, nullable=False)
    num_positions = Column(Integer, default=0)
    daily_pnl = Column(Float, nullable=True)
    daily_pnl_pct = Column(Float, nullable=True)
    total_pnl = Column(Float, nullable=True)
    total_pnl_pct = Column(Float, nullable=True)
    positions_json = Column(JSON, nullable=True)


# ============================================================
# DATABASE OPERATIONS
# ============================================================

class Database:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized: {database_url}")

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ── TRADES ──────────────────────────────────────────────────────

    def create_trade(self, **kwargs) -> Trade:
        with self.get_session() as session:
            trade = Trade(**kwargs)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade

    def update_trade(self, trade_id: int, **kwargs) -> Optional[Trade]:
        with self.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in kwargs.items():
                    setattr(trade, key, value)
                session.commit()
            return trade

    def get_open_trades(self) -> List[Trade]:
        with self.get_session() as session:
            return session.query(Trade).filter(Trade.status == "open").all()

    def get_open_trade_by_symbol(self, symbol: str) -> Optional[Trade]:
        """The current open position for an underlying (one position per symbol)."""
        with self.get_session() as session:
            return session.query(Trade).filter(
                Trade.symbol == symbol, Trade.status == "open"
            ).order_by(Trade.entry_time.desc()).first()

    def get_open_trade_by_contract(self, contract_symbol: str) -> Optional[Trade]:
        with self.get_session() as session:
            return session.query(Trade).filter(
                Trade.contract_symbol == contract_symbol, Trade.status == "open"
            ).first()

    def get_all_trades(self, limit: int = 100) -> List[Trade]:
        with self.get_session() as session:
            return session.query(Trade).order_by(Trade.created_at.desc()).limit(limit).all()

    def get_closed_trades(self, days: int = 30) -> List[Trade]:
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            return session.query(Trade).filter(
                Trade.status == "closed", Trade.exit_time >= cutoff
            ).all()

    def get_daily_pnl(self) -> float:
        with self.get_session() as session:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            trades_today = session.query(Trade).filter(
                Trade.status == "closed", Trade.exit_time >= today_start,
            ).all()
            return sum(t.pnl or 0.0 for t in trades_today)

    # ── SIGNAL LOG ──────────────────────────────────────────────────

    def log_signal(self, **kwargs) -> SignalLog:
        with self.get_session() as session:
            row = SignalLog(**kwargs)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def get_recent_signals(self, limit: int = 100) -> List[SignalLog]:
        with self.get_session() as session:
            return session.query(SignalLog).order_by(SignalLog.timestamp.desc()).limit(limit).all()

    def get_latest_signal_per_symbol(self) -> List[SignalLog]:
        """Most recent SignalLog row for each symbol (for the dashboard Signals page)."""
        with self.get_session() as session:
            rows = session.query(SignalLog).order_by(SignalLog.timestamp.desc()).limit(1000).all()
        seen: Dict[str, SignalLog] = {}
        for r in rows:
            if r.symbol not in seen:
                seen[r.symbol] = r
        return list(seen.values())

    # ── PORTFOLIO ────────────────────────────────────────────────────

    def log_portfolio_snapshot(self, **kwargs) -> PortfolioSnapshot:
        with self.get_session() as session:
            snap = PortfolioSnapshot(**kwargs)
            session.add(snap)
            session.commit()
            return snap

    def get_portfolio_history(self, days: int = 30) -> List[PortfolioSnapshot]:
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            return session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.timestamp >= cutoff
            ).order_by(PortfolioSnapshot.timestamp.asc()).all()

    def get_latest_portfolio(self) -> Optional[PortfolioSnapshot]:
        with self.get_session() as session:
            return session.query(PortfolioSnapshot).order_by(
                PortfolioSnapshot.timestamp.desc()
            ).first()

    # ── PERFORMANCE METRICS ──────────────────────────────────────────

    def compute_performance_metrics(self, days: int = 90) -> Dict[str, Any]:
        closed = self.get_closed_trades(days=days)
        if not closed:
            return {}
        wins = [t for t in closed if t.pnl and t.pnl > 0]
        losses = [t for t in closed if t.pnl and t.pnl <= 0]
        win_rate = len(wins) / len(closed) if closed else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 and wins else 0
        total_pnl = sum(t.pnl for t in closed if t.pnl)

        exit_reasons: Dict[str, int] = {}
        for t in closed:
            reason = t.exit_reason or "unknown"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            "total_trades": len(closed),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "exit_reasons": exit_reasons,
        }
