"""
AI Investor Agent - FastAPI Backend v2.0
REST API serving the dashboard with live data.
Key fix: risk settings now save to SQLite so scheduler picks them up without restart.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from agent.config import config
from agent.database import Database
from agent.executor import Executor
from agent.feedback.tuner import ParameterTuner
from agent.feedback.analyzer import PerformanceAnalyzer

app = FastAPI(title="AI Investor Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(config.database_url)
db.ensure_risk_settings_exist()
executor = Executor()
tuner = ParameterTuner(db)
analyzer = PerformanceAnalyzer(db)


# ── Pydantic models ───────────────────────────────────────────────────

class RiskUpdateRequest(BaseModel):
    max_positions: Optional[int] = None
    max_position_pct: Optional[float] = None
    cash_reserve_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    portfolio_drawdown_limit_pct: Optional[float] = None
    min_premium: Optional[float] = None
    max_premium: Optional[float] = None


class WatchlistUpdateRequest(BaseModel):
    watchlist: List[str]


# ── Serializers ───────────────────────────────────────────────────────

def trade_to_dict(t) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "side": t.side,
        "contract_symbol": t.contract_symbol,
        "option_type": t.option_type,
        "strategy_type": t.strategy_type,
        "strike": t.strike,
        "expiration": t.expiration,
        "dte_at_entry": t.dte_at_entry,
        "iv_at_entry": t.iv_at_entry,
        "contracts": t.contracts,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "premium_paid": t.premium_paid,
        "entry_time": str(t.entry_time) if t.entry_time else None,
        "exit_time": str(t.exit_time) if t.exit_time else None,
        "status": t.status,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "exit_reason": t.exit_reason,
        "strategy_version": t.strategy_version,
    }


def decision_to_dict(d) -> dict:
    return {
        "id": d.id,
        "timestamp": str(d.timestamp),
        "symbol": d.symbol,
        "action": d.action,
        "confidence": d.confidence,
        "reasoning": d.reasoning,
        "signal_summary": d.signal_summary,
        "strategy_mode": d.strategy_mode,
        "target_dte": d.target_dte,
        "suggested_pct": d.suggested_pct,
        "executed": d.executed,
        "execution_blocked_reason": d.execution_blocked_reason,
        "strategy_version": d.strategy_version,
    }


def portfolio_to_dict(p) -> dict:
    return {
        "timestamp": str(p.timestamp),
        "equity": p.equity,
        "cash": p.cash,
        "invested_value": p.invested_value,
        "num_positions": p.num_positions,
        "daily_pnl": p.daily_pnl,
        "daily_pnl_pct": p.daily_pnl_pct,
        "total_pnl": p.total_pnl,
        "total_pnl_pct": p.total_pnl_pct,
        "drawdown_pct": p.drawdown_pct,
        "circuit_breaker_active": p.circuit_breaker_active,
        "positions": p.positions_json,
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "timestamp": datetime.utcnow().isoformat()}


@app.get("/account")
def get_account():
    try:
        return executor.get_account()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/portfolio/latest")
def get_latest_portfolio():
    snap = db.get_latest_portfolio()
    return portfolio_to_dict(snap) if snap else {}


@app.get("/portfolio/history")
def get_portfolio_history(days: int = 30):
    return [portfolio_to_dict(p) for p in db.get_portfolio_history(days=days)]


@app.get("/positions")
def get_positions():
    return executor.get_positions()


@app.get("/positions/options")
def get_option_positions():
    """Return only open options positions."""
    return executor.get_option_positions()


@app.get("/trades")
def get_trades(limit: int = 100):
    return [trade_to_dict(t) for t in db.get_all_trades(limit=limit)]


@app.get("/trades/open")
def get_open_trades():
    return [trade_to_dict(t) for t in db.get_open_trades()]


@app.get("/decisions")
def get_decisions(limit: int = 50):
    return [decision_to_dict(d) for d in db.get_recent_decisions(limit=limit)]


@app.get("/performance")
def get_performance(days: int = 30):
    return db.compute_performance_metrics(days=days)


@app.get("/signals/{symbol}")
def get_latest_signals(symbol: str):
    snap = db.get_latest_signals(symbol.upper())
    if not snap:
        return {}
    return {
        "symbol": snap.symbol,
        "timestamp": str(snap.timestamp),
        "rsi": snap.rsi,
        "macd": snap.macd,
        "macd_crossover": snap.macd_crossover,
        "ema_9": snap.ema_9,
        "ema_21": snap.ema_21,
        "ema_50": snap.ema_50,
        "above_ema_trend": snap.above_ema_trend,
        "bollinger_pct": snap.bollinger_pct,
        "atr": snap.atr,
        "volume_ratio": snap.volume_ratio,
        "sentiment_score": snap.sentiment_score,
        "news_count": snap.news_count,
        "sentiment_summary": snap.sentiment_summary,
        "sentiment_from_cache": snap.sentiment_from_cache,
        "vix_level": snap.vix_level,
        "market_regime": snap.market_regime,
        "iv_rank": snap.iv_rank,
        "iv_regime": snap.iv_regime,
        "put_call_ratio": snap.put_call_ratio,
        "earnings_blackout": snap.earnings_blackout,
        "strategy_mode": snap.strategy_mode,
        "close_price": snap.close_price,
        "technical_score": snap.technical_score,
        "options_flow_score": snap.options_flow_score,
        "overall_confidence": snap.overall_confidence,
    }


@app.get("/risk/settings")
def get_risk_settings():
    """Current risk parameters (always reads from DB — source of truth)."""
    settings = db.get_risk_settings()
    if settings:
        latest_portfolio = db.get_latest_portfolio()
        circuit_breaker = latest_portfolio.circuit_breaker_active if latest_portfolio else False
        return {
            "initial_capital": config.risk.initial_capital,
            "max_positions": settings.max_positions,
            "max_position_pct": settings.max_position_pct,
            "cash_reserve_pct": settings.cash_reserve_pct,
            "daily_loss_limit_pct": settings.daily_loss_limit_pct,
            "portfolio_drawdown_limit_pct": settings.portfolio_drawdown_limit_pct,
            "min_premium": settings.min_premium,
            "max_premium": settings.max_premium,
            "circuit_breaker_active": circuit_breaker,
            "updated_at": str(settings.updated_at),
        }
    return {"error": "Risk settings not found"}


@app.post("/risk/reset-circuit-breaker")
def reset_circuit_breaker():
    """Clear the circuit breaker flag so the agent resumes trading."""
    db.reset_circuit_breaker_flag()
    logger.warning("Circuit breaker reset via dashboard")
    return {"status": "reset"}


@app.post("/risk/settings")
def update_risk_settings(req: RiskUpdateRequest):
    """
    Update risk parameters.
    Saves to SQLite — scheduler reads these on next cycle without restart.
    """
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No values provided to update")

    db.save_risk_settings(**updates)
    logger.info(f"Risk settings updated via dashboard: {updates}")
    return {"status": "updated", "settings": get_risk_settings()}


@app.get("/watchlist")
def get_watchlist():
    return {"watchlist": config.strategy.watchlist}


@app.post("/watchlist")
def update_watchlist(req: WatchlistUpdateRequest):
    config.strategy.watchlist = [s.upper() for s in req.watchlist]
    return {"status": "updated", "watchlist": config.strategy.watchlist}


@app.get("/trading-mode")
def get_trading_mode():
    return {
        "paper": config.alpaca.paper,
        "mode": "paper" if config.alpaca.paper else "live",
        "label": "📄 Paper Trading" if config.alpaca.paper else "💰 Live Trading",
        "initial_capital": config.risk.initial_capital,
    }


@app.get("/orders/recent")
def get_recent_orders(limit: int = 20):
    return executor.get_recent_orders(limit=limit)


@app.get("/activity")
def get_activity(days: int = 7, limit: int = 100, filter: str = "all"):
    """
    Unified activity timeline: GPT-4o decisions + trade entries + exits.
    filter: "all" | "decisions" | "trades"
    Sorted newest-first.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)
    events = []

    if filter in ("all", "decisions"):
        for d in db.get_decisions_since(cutoff):
            events.append({
                "type": "decision",
                "timestamp": str(d.timestamp),
                "symbol": d.symbol,
                "action": d.action,
                "confidence": d.confidence,
                "reasoning": d.reasoning,
                "signal_scores": d.signal_summary,
                "executed": d.executed,
                "execution_blocked_reason": d.execution_blocked_reason,
                "strategy_mode": d.strategy_mode,
            })

    if filter in ("all", "trades"):
        for t in db.get_trades_since(cutoff):
            if t.entry_time:
                events.append({
                    "type": "entry",
                    "timestamp": str(t.entry_time),
                    "symbol": t.symbol,
                    "contract_symbol": t.contract_symbol,
                    "option_type": t.option_type,
                    "strategy_type": t.strategy_type,
                    "strike": t.strike,
                    "expiration": t.expiration,
                    "dte_at_entry": t.dte_at_entry,
                    "contracts": t.contracts,
                    "premium_paid": t.premium_paid,
                    "entry_price": t.entry_price,
                    "trade_id": t.id,
                })
            if t.exit_time and t.status in ("closed", "expired"):
                events.append({
                    "type": "exit",
                    "timestamp": str(t.exit_time),
                    "symbol": t.symbol,
                    "contract_symbol": t.contract_symbol,
                    "option_type": t.option_type,
                    "strategy_type": t.strategy_type,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_price": t.exit_price,
                    "premium_paid": t.premium_paid,
                    "trade_id": t.id,
                })

    events.sort(key=lambda x: x["timestamp"], reverse=True)
    return {
        "events": events[:limit],
        "total": len(events),
        "days": days,
    }


@app.get("/daily-pnl")
def get_daily_pnl():
    """Today's realized P&L from closed trades."""
    pnl = db.get_daily_pnl()
    return {
        "daily_pnl": round(pnl, 2),
        "daily_pnl_pct": round(pnl / config.risk.initial_capital, 4) if config.risk.initial_capital else 0,
    }


# ── FEEDBACK & SELF-IMPROVEMENT ──────────────────────────────────────────────

@app.get("/feedback/latest-report")
def get_latest_report(review_type: Optional[str] = None):
    """Return the most recent weekly or monthly feedback report."""
    import json
    report = db.get_latest_feedback_report(review_type=review_type)
    if not report:
        return {"message": "No feedback reports yet. First report generates Friday after market close."}
    return {
        "id": report.id,
        "review_type": report.review_type,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "period_days": report.period_days,
        "total_trades": report.total_trades,
        "assessment": report.assessment,
        "top_issues": json.loads(report.recommendations_json or "[]"),
        "recommendations": json.loads(report.recommendations_json or "[]"),
        "applied_changes": json.loads(report.applied_changes_json or "[]"),
        "red_flags": json.loads(report.red_flags_json or "[]"),
        "parameters_changed": report.parameters_changed,
        "alpha_decay_status": report.alpha_decay_status,
        "overview": json.loads(report.overview_json or "{}"),
    }


@app.get("/feedback/reports")
def get_all_reports(limit: int = 12):
    """Return the last N feedback reports (weekly + monthly combined)."""
    import json
    reports = db.get_feedback_reports(limit=limit)
    return [
        {
            "id": r.id,
            "review_type": r.review_type,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "period_days": r.period_days,
            "total_trades": r.total_trades,
            "parameters_changed": r.parameters_changed,
            "applied_changes_count": len(json.loads(r.applied_changes_json or "[]")),
            "red_flags_count": len(json.loads(r.red_flags_json or "[]")),
            "alpha_decay_status": r.alpha_decay_status,
            "assessment": r.assessment,
        }
        for r in reports
    ]


@app.get("/feedback/parameter-history")
def get_parameter_history(limit: int = 50):
    """Full audit trail of every parameter change the system has made."""
    history = db.get_parameter_history(limit=limit)
    return [
        {
            "id": h.id,
            "parameter": h.parameter_name,
            "old_value": h.old_value,
            "new_value": h.new_value,
            "reasoning": h.reasoning,
            "changed_at": h.changed_at.isoformat() if h.changed_at else None,
        }
        for h in history
    ]


@app.get("/feedback/analysis")
def get_live_analysis(days: int = 30):
    """Run a fresh performance analysis right now (same data the monthly review uses)."""
    return analyzer.run_full_analysis(days=days)


@app.post("/feedback/run-weekly")
def trigger_weekly_review():
    """Manually trigger a weekly review (useful for testing)."""
    try:
        report = tuner.run_weekly_review()
        return {"status": "complete", "assessment": report.get("ai_assessment", ""), "red_flags": report.get("red_flags", [])}
    except Exception as e:
        raise HTTPException(500, f"Review failed: {e}")


@app.post("/feedback/run-monthly")
def trigger_monthly_review():
    """Manually trigger a full monthly review with parameter adjustment."""
    try:
        report = tuner.run_monthly_review()
        return {
            "status": "complete",
            "parameters_changed": report.get("parameters_changed", False),
            "applied_changes": report.get("applied_changes", []),
            "assessment": report.get("ai_assessment", ""),
            "red_flags": report.get("red_flags", []),
        }
    except Exception as e:
        raise HTTPException(500, f"Review failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.dashboard_api_host, port=config.dashboard_api_port)
