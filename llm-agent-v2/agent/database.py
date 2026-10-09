"""
AI Investor Agent - Database Layer v2.0
SQLite schema for options trading.
Key additions vs v1.0:
  - Trade table: options columns (contract_symbol, strike, expiry, DTE, IV, premium, peak_value)
  - RiskSettings table: cross-process persistence so dashboard changes reach the scheduler
  - get_daily_pnl(): real daily P&L so the scheduler no longer hardcodes 0.0
  - get_or_create_risk_settings() / save_risk_settings()
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, Boolean, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger

Base = declarative_base()


# ============================================================
# MODELS
# ============================================================

class Trade(Base):
    """Every options trade the agent makes — open and closed."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)   # Underlying, e.g. "AAPL"
    side = Column(String(4), nullable=False)                  # "buy"

    # Options-specific fields
    contract_symbol = Column(String(25), nullable=True)       # OCC symbol e.g. "AAPL240119C00150000"
    option_type = Column(String(4), nullable=True)            # "CALL" or "PUT"
    strategy_type = Column(String(15), nullable=True)         # "BUY_CALL"|"BUY_PUT"|"BULL_SPREAD"|"BEAR_SPREAD"
    strike = Column(Float, nullable=True)
    expiration = Column(String(10), nullable=True)            # "YYYY-MM-DD"
    dte_at_entry = Column(Integer, nullable=True)
    iv_at_entry = Column(Float, nullable=True)
    contracts = Column(Integer, default=1)                    # Number of contracts

    # Pricing
    entry_price = Column(Float, nullable=True)                # Premium per share at entry
    exit_price = Column(Float, nullable=True)                 # Premium per share at exit
    premium_paid = Column(Float, nullable=True)               # Total cost (entry_price × 100 × contracts)
    peak_value = Column(Float, nullable=True)                 # Highest total value reached (for trailing logic)

    # Timing
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)

    # Outcome
    status = Column(String(10), default="open")               # "open" | "closed" | "expired"
    pnl = Column(Float, nullable=True)                        # Dollar P&L
    pnl_pct = Column(Float, nullable=True)                    # % gain/loss on premium
    exit_reason = Column(String(50), nullable=True)           # "stop_loss"|"profit_target"|"time_exit"|"sentiment_flip"

    # Sentiment decay tracking (added v2.1)
    catalyst_type = Column(String(20), nullable=True)          # "fundamental"|"momentum"|"sentiment_buzz"
    sentiment_shelf_life_days = Column(Integer, nullable=True) # days before re-checking sentiment
    sentiment_score_at_entry = Column(Float, nullable=True)    # raw -1..+1 score at entry time

    # Credit-spread tracking (added 2026-06-10, MASTER_STRATEGY.md CORE).
    # For a SOLD credit spread: contract_symbol = SHORT leg, long_leg_symbol = protective long leg.
    # credit_received is what we collected (positive); spread_width is the strike width × 100.
    long_leg_symbol = Column(String(25), nullable=True)        # OCC of the protective long leg
    credit_received = Column(Float, nullable=True)             # total $ credit collected at entry
    spread_width = Column(Float, nullable=True)                # $ width per spread (width_pts × 100)

    # Metadata
    alpaca_order_id = Column(String(50), nullable=True)
    strategy_version = Column(String(10), default="v2.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalSnapshot(Base):
    """Snapshot of all signals when a decision was made."""
    __tablename__ = "signal_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Technical
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    macd_crossover = Column(Boolean, nullable=True)
    ema_9 = Column(Float, nullable=True)
    ema_21 = Column(Float, nullable=True)
    ema_50 = Column(Float, nullable=True)
    above_ema_trend = Column(Boolean, nullable=True)
    bollinger_pct = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)

    # Sentiment
    sentiment_score = Column(Float, nullable=True)
    news_count = Column(Integer, nullable=True)
    sentiment_summary = Column(Text, nullable=True)
    sentiment_from_cache = Column(Boolean, default=False)

    # Macro
    vix_level = Column(Float, nullable=True)
    market_regime = Column(String(20), nullable=True)

    # Options Flow (new)
    iv_rank = Column(Float, nullable=True)
    iv_regime = Column(String(15), nullable=True)
    put_call_ratio = Column(Float, nullable=True)
    earnings_blackout = Column(Boolean, default=False)
    strategy_mode = Column(String(15), nullable=True)

    # Price
    close_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)

    # Composite
    technical_score = Column(Float, nullable=True)
    options_flow_score = Column(Float, nullable=True)
    overall_confidence = Column(Float, nullable=True)


class AgentDecision(Base):
    """The LLM's reasoning and final decision for each scan."""
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(15), nullable=False)               # "BUY_CALL"|"BUY_PUT"|etc.
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    signal_summary = Column(JSON, nullable=True)
    strategy_mode = Column(String(15), nullable=True)
    target_dte = Column(Integer, nullable=True)
    suggested_pct = Column(Float, nullable=True)
    executed = Column(Boolean, default=False)
    execution_blocked_reason = Column(String(200), nullable=True)
    trade_id = Column(Integer, nullable=True)
    strategy_version = Column(String(10), default="v2.0")


class PortfolioSnapshot(Base):
    """Portfolio state every scan cycle — enables equity curve and drawdown tracking."""
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
    peak_equity = Column(Float, nullable=True)
    drawdown_pct = Column(Float, nullable=True)
    positions_json = Column(JSON, nullable=True)
    circuit_breaker_active = Column(Boolean, default=False)


class RiskSettings(Base):
    """
    Persisted risk parameters — written by the dashboard API, read by the scheduler.
    Single row (id=1) that both processes share via SQLite.
    This fixes the cross-process sync bug from v1.0.
    """
    __tablename__ = "risk_settings"

    id = Column(Integer, primary_key=True, default=1)
    max_positions = Column(Integer, default=4)
    max_position_pct = Column(Float, default=0.08)
    cash_reserve_pct = Column(Float, default=0.20)
    daily_loss_limit_pct = Column(Float, default=0.03)
    portfolio_drawdown_limit_pct = Column(Float, default=0.15)
    min_premium = Column(Float, default=75.0)
    max_premium = Column(Float, default=800.0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class FeedbackReport(Base):
    """Stores every weekly/monthly AI review report."""
    __tablename__ = "feedback_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_type = Column(String(10), nullable=False)          # "weekly" | "monthly"
    period_days = Column(Integer, default=30)
    total_trades = Column(Integer, default=0)
    assessment = Column(Text, nullable=True)                  # GPT-4o executive summary
    recommendations_json = Column(Text, nullable=True)        # JSON list of recommendations
    applied_changes_json = Column(Text, nullable=True)        # JSON list of applied changes
    red_flags_json = Column(Text, nullable=True)              # JSON list of red flags
    parameters_changed = Column(Boolean, default=False)
    alpha_decay_status = Column(String(20), nullable=True)    # IMPROVING|STABLE|DECAYING
    overview_json = Column(Text, nullable=True)               # Snapshot of overview metrics
    created_at = Column(DateTime, default=datetime.utcnow)


class ParameterSnapshot(Base):
    """Audit trail of every parameter change made by the feedback system."""
    __tablename__ = "parameter_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_name = Column(String(50), nullable=False)
    old_value = Column(String(50), nullable=True)
    new_value = Column(String(50), nullable=False)
    reasoning = Column(Text, nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow)


class NewsCache(Base):
    """Cached news items — prevents redundant Finnhub calls."""
    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    headline = Column(Text, nullable=False)
    source = Column(String(100), nullable=True)
    url = Column(String(500), nullable=True)
    published_at = Column(DateTime, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================
# DATABASE OPERATIONS
# ============================================================

class Database:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self._run_migrations()
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized: {database_url}")

    def _run_migrations(self):
        """Add new columns to existing tables without destroying existing data.
        SQLite raises an error on duplicate ADD COLUMN — we catch and skip those."""
        migrations = [
            "ALTER TABLE trades ADD COLUMN catalyst_type VARCHAR(20)",
            "ALTER TABLE trades ADD COLUMN sentiment_shelf_life_days INTEGER",
            "ALTER TABLE trades ADD COLUMN sentiment_score_at_entry FLOAT",
            "ALTER TABLE trades ADD COLUMN long_leg_symbol VARCHAR(25)",
            "ALTER TABLE trades ADD COLUMN credit_received FLOAT",
            "ALTER TABLE trades ADD COLUMN spread_width FLOAT",
        ]
        with self.engine.connect() as conn:
            for sql in migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                except Exception:
                    pass  # column already exists

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

    def update_trade_peak(self, trade_id: int, current_value: float):
        """Update peak_value if current_value exceeds it. Used for trailing stop logic."""
        with self.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade and (trade.peak_value is None or current_value > trade.peak_value):
                trade.peak_value = current_value
                session.commit()

    def get_open_trades(self) -> List[Trade]:
        with self.get_session() as session:
            return session.query(Trade).filter(Trade.status == "open").all()

    def get_open_trade_by_contract(self, contract_symbol: str) -> Optional[Trade]:
        with self.get_session() as session:
            return session.query(Trade).filter(
                Trade.contract_symbol == contract_symbol,
                Trade.status == "open"
            ).first()

    def get_all_trades(self, limit: int = 100) -> List[Trade]:
        with self.get_session() as session:
            return session.query(Trade).order_by(Trade.created_at.desc()).limit(limit).all()

    def get_closed_trades(self, days: int = 30) -> List[Trade]:
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            return session.query(Trade).filter(
                Trade.status == "closed",
                Trade.exit_time >= cutoff
            ).all()

    def get_daily_pnl(self) -> float:
        """
        Sum P&L of all trades closed today (UTC).
        This is what the scheduler passes to check_portfolio_health — fixes the 0.0 hardcode bug.
        """
        with self.get_session() as session:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            trades_today = session.query(Trade).filter(
                Trade.status == "closed",
                Trade.exit_time >= today_start,
            ).all()
            return sum(t.pnl or 0.0 for t in trades_today)

    # ── DECISIONS ────────────────────────────────────────────────────

    def log_decision(self, **kwargs) -> AgentDecision:
        with self.get_session() as session:
            decision = AgentDecision(**kwargs)
            session.add(decision)
            session.commit()
            session.refresh(decision)
            return decision

    def get_recent_decisions(self, limit: int = 50) -> List[AgentDecision]:
        with self.get_session() as session:
            return session.query(AgentDecision).order_by(
                AgentDecision.timestamp.desc()
            ).limit(limit).all()

    def get_decisions_since(self, cutoff: datetime, limit: int = 500) -> List[AgentDecision]:
        with self.get_session() as session:
            return session.query(AgentDecision).filter(
                AgentDecision.timestamp >= cutoff
            ).order_by(AgentDecision.timestamp.desc()).limit(limit).all()

    def get_trades_since(self, cutoff: datetime, limit: int = 200) -> List[Trade]:
        with self.get_session() as session:
            return session.query(Trade).filter(
                Trade.created_at >= cutoff
            ).order_by(Trade.created_at.desc()).limit(limit).all()

    # ── SIGNALS ──────────────────────────────────────────────────────

    def log_signal_snapshot(self, **kwargs) -> SignalSnapshot:
        with self.get_session() as session:
            snap = SignalSnapshot(**kwargs)
            session.add(snap)
            session.commit()
            return snap

    def get_latest_signals(self, symbol: str) -> Optional[SignalSnapshot]:
        with self.get_session() as session:
            return session.query(SignalSnapshot).filter(
                SignalSnapshot.symbol == symbol
            ).order_by(SignalSnapshot.timestamp.desc()).first()

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

    def reset_circuit_breaker_flag(self):
        """Write circuit_breaker_active=False into the latest portfolio snapshot."""
        with self.get_session() as session:
            snap = session.query(PortfolioSnapshot).order_by(
                PortfolioSnapshot.timestamp.desc()
            ).first()
            if snap:
                snap.circuit_breaker_active = False
                session.commit()

    # ── RISK SETTINGS (cross-process persistence) ─────────────────

    def get_risk_settings(self) -> Optional[RiskSettings]:
        """Returns the single RiskSettings row, or None if not yet created."""
        with self.get_session() as session:
            return session.query(RiskSettings).filter(RiskSettings.id == 1).first()

    def save_risk_settings(self, **kwargs) -> RiskSettings:
        """
        Upsert risk settings (always row id=1).
        Called by the dashboard API when user changes a slider.
        The scheduler reads this on every cycle to stay in sync.
        """
        with self.get_session() as session:
            settings = session.query(RiskSettings).filter(RiskSettings.id == 1).first()
            if settings is None:
                settings = RiskSettings(id=1)
                session.add(settings)
            for key, value in kwargs.items():
                if hasattr(settings, key) and value is not None:
                    setattr(settings, key, value)
            settings.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(settings)
            return settings

    def ensure_risk_settings_exist(self):
        """
        Sync risk settings from config/.env into DB on every startup.
        Config wins on restart — ensures .env changes always take effect.
        Dashboard overrides persist during a session but are overwritten on next restart.
        """
        from agent.config import config
        cfg = config.risk
        self.save_risk_settings(
            max_positions=cfg.max_positions,
            max_position_pct=cfg.max_position_pct,
            cash_reserve_pct=cfg.cash_reserve_pct,
            daily_loss_limit_pct=cfg.daily_loss_limit_pct,
            portfolio_drawdown_limit_pct=cfg.portfolio_drawdown_limit_pct,
            min_premium=cfg.min_premium,
            max_premium=cfg.max_premium,
        )
        logger.info(
            f"Risk settings synced from config: max_positions={cfg.max_positions}, "
            f"daily_loss_limit={cfg.daily_loss_limit_pct:.0%}"
        )

    # ── FEEDBACK REPORTS ────────────────────────────────────────────

    def get_latest_feedback_report(self, review_type: Optional[str] = None) -> Optional["FeedbackReport"]:
        with self.get_session() as session:
            q = session.query(FeedbackReport).order_by(FeedbackReport.created_at.desc())
            if review_type:
                q = q.filter(FeedbackReport.review_type == review_type)
            return q.first()

    def get_feedback_reports(self, limit: int = 12) -> List["FeedbackReport"]:
        with self.get_session() as session:
            return session.query(FeedbackReport).order_by(
                FeedbackReport.created_at.desc()
            ).limit(limit).all()

    def get_parameter_history(self, limit: int = 50) -> List["ParameterSnapshot"]:
        with self.get_session() as session:
            return session.query(ParameterSnapshot).order_by(
                ParameterSnapshot.changed_at.desc()
            ).limit(limit).all()

    # ── PERFORMANCE METRICS ──────────────────────────────────────────

    def compute_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
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

        strategy_breakdown: Dict[str, Dict] = {}
        for t in closed:
            stype = t.strategy_type or "unknown"
            if stype not in strategy_breakdown:
                strategy_breakdown[stype] = {"count": 0, "pnl": 0.0, "wins": 0}
            strategy_breakdown[stype]["count"] += 1
            strategy_breakdown[stype]["pnl"] += t.pnl or 0
            if t.pnl and t.pnl > 0:
                strategy_breakdown[stype]["wins"] += 1

        return {
            "total_trades": len(closed),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "exit_reasons": exit_reasons,
            "strategy_breakdown": strategy_breakdown,
        }
