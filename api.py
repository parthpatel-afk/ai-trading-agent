"""
AI Investor v2 — FastAPI Backend
Serves the dashboard with live data from the EMA crossover bot.
Read-only: the strategy is fixed, so there are no settings to write.
"""
import base64
import io
import json
import os
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agent.config import config
from agent.database import Database
from agent.executor import Executor
from agent.logging_setup import setup_logging
from agent.paths import db_path, logs_dir

# Where the "Send Logs" button uploads to (an Apps Script drop box, configured through env vars).
LOGS_UPLOAD_URL = os.getenv("LOGS_UPLOAD_URL", "")
LOGS_UPLOAD_SECRET = os.getenv("LOGS_UPLOAD_SECRET", "")

setup_logging()

app = FastAPI(title="Trend-Buyer v1 API", version="trendbuyer-v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(config.database_url)
executor = Executor()


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
    }


def signal_to_dict(s) -> dict:
    return {
        "id": s.id,
        "timestamp": str(s.timestamp),
        "symbol": s.symbol,
        "close": s.close,
        "ema_fast": s.ema_fast,
        "ema_slow": s.ema_slow,
        "fast_above_slow": s.fast_above_slow,
        "cross": s.cross,
        "cross_age_days": s.cross_age_days,
        "fresh": s.fresh,
        "action": s.action,
        "reason": s.reason,
        "executed": s.executed,
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
        "positions": p.positions_json,
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "ema-v1", "timestamp": datetime.utcnow().isoformat()}


@app.get("/account")
def get_account():
    try:
        return executor.get_account()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/strategy")
def get_strategy():
    """Static description of the running strategy (for the dashboard header)."""
    return {
        "name": "Trend-Buyer v1 (9/21 EMA + filters, deep-ITM)",
        "timeframe": "Daily",
        "watchlist": config.strategy.watchlist,
        "ema_fast": config.strategy.ema_fast,
        "ema_slow": config.strategy.ema_slow,
        "target_delta": config.options.target_delta,
        "target_dte": config.options.target_dte,
        "max_dte": config.options.max_dte,
        "risk_per_trade_pct": config.account.risk_per_trade_pct,
        "max_positions": config.account.max_positions,
        "time_exit_dte": config.exits.time_exit_dte,
        "initial_capital": config.account.initial_capital,
        "paper": config.alpaca.paper,
    }


@app.get("/portfolio/latest")
def get_latest_portfolio():
    snap = db.get_latest_portfolio()
    return portfolio_to_dict(snap) if snap else {}


@app.get("/portfolio/history")
def get_portfolio_history(days: int = 30):
    return [portfolio_to_dict(p) for p in db.get_portfolio_history(days=days)]


@app.get("/trades")
def get_trades(limit: int = 100):
    return [trade_to_dict(t) for t in db.get_all_trades(limit=limit)]


@app.get("/trades/open")
def get_open_trades():
    return [trade_to_dict(t) for t in db.get_open_trades()]


@app.get("/positions")
def get_positions():
    try:
        return executor.get_positions()
    except Exception as e:
        raise HTTPException(500, str(e))


def _enrich_with_mid(positions: list) -> list:
    """Add a fair MID price + mid-based P&L to each option position (review item 8).
    Alpaca's own mark is the conservative bid, which looks alarmingly red right after a
    buy; the mid is a more representative value. We surface both so the dashboard can
    show the mid. Quotes are cached in the executor so this stays cheap on dashboard polls."""
    for p in positions:
        try:
            q = executor.get_option_quote(p.get("symbol"))
            if not q or q.get("mid", 0) <= 0:
                continue
            qty = abs(float(p.get("qty", 0)))
            mid = float(q["mid"])
            mid_mv = round(mid * 100 * qty, 2)
            cost = float(p.get("avg_entry_price", 0)) * 100 * qty
            p["bid"] = q.get("bid")
            p["ask"] = q.get("ask")
            p["mid_price"] = mid
            p["mid_market_value"] = mid_mv
            p["mid_unrealized_pl"] = round(mid_mv - cost, 2)
            p["mid_unrealized_plpc"] = round((mid_mv - cost) / cost, 4) if cost else 0
        except Exception:
            continue
    return positions


@app.get("/positions/options")
def get_option_positions():
    try:
        return _enrich_with_mid(executor.get_option_positions())
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/daily-pnl")
def get_daily_pnl():
    snap = db.get_latest_portfolio()
    if not snap:
        return {"daily_pnl": 0, "daily_pnl_pct": 0}
    return {"daily_pnl": snap.daily_pnl, "daily_pnl_pct": snap.daily_pnl_pct}


@app.get("/signals")
def get_signals():
    """Latest EMA evaluation per symbol (the Signals page)."""
    rows = db.get_latest_signal_per_symbol()
    rows.sort(key=lambda s: s.symbol)
    return [signal_to_dict(s) for s in rows]


@app.get("/signals/recent")
def get_recent_signals(limit: int = 100):
    return [signal_to_dict(s) for s in db.get_recent_signals(limit=limit)]


@app.get("/performance")
def get_performance(days: int = 90):
    return db.compute_performance_metrics(days=days)


@app.post("/send-logs")
def send_logs():
    """Zip the DB + logs and upload them to the drop box set in LOGS_UPLOAD_URL."""
    if not LOGS_UPLOAD_URL:
        raise HTTPException(503, "log upload is not configured (set LOGS_UPLOAD_URL and LOGS_UPLOAD_SECRET)")
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            dbp = db_path()
            if dbp.exists():
                z.write(str(dbp), arcname=dbp.name)
            for f in logs_dir().glob("*.log"):
                z.write(str(f), arcname=f"logs/{f.name}")
        filename = f"AIv2_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        payload = json.dumps({
            "secret": LOGS_UPLOAD_SECRET,
            "filename": filename,
            "data": base64.b64encode(buf.getvalue()).decode(),
        }).encode()
        req = urllib.request.Request(LOGS_UPLOAD_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = r.read().decode()[:120]
        if "ok" in resp.lower():
            logger.info(f"Logs sent to support: {filename}")
            return {"status": "sent", "filename": filename}
        logger.warning(f"Log upload rejected: {resp}")
        raise HTTPException(502, f"upload rejected: {resp}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"send-logs failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/version")
def get_version():
    return {"version": "trendbuyer-v1"}


@app.post("/shutdown")
def shutdown_app():
    """Stop the whole app (the dashboard 'Quit Bot' button calls this)."""
    logger.info("Shutdown requested from the dashboard — exiting.")

    def _die():
        time.sleep(0.4)  # let the HTTP response flush first
        os._exit(0)      # terminate the entire process (API + dashboard + bot threads)

    threading.Thread(target=_die, daemon=True).start()
    return {"status": "stopping"}


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API on {config.dashboard_api_host}:{config.dashboard_api_port}")
    uvicorn.run(app, host=config.dashboard_api_host, port=config.dashboard_api_port)
