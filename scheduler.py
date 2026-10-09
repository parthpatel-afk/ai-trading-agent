"""
Trend-Buyer v1 — Main Agent Loop (long deep-ITM options, daily trend-follower)
Entry point: python scheduler.py

Two jobs per trading day:
  • 09:35 ET  run_open_routine   — flips + new entries (EMA fresh cross → filters → deep-ITM buy)
  • 15:50 ET  run_close_monitor  — exits on the day's (near-)close: time-exit, hard stop,
                                   partial profit-take (arms the trail), Chandelier trail
Plus a portfolio snapshot every 15 min for the dashboard.

Mechanical and deterministic (no AI — that's the planned Phase 2). Every exit is judged on
the UNDERLYING's daily close with a 2-read confirmation (the hard-won ledger rule).
"""
import signal
import sys
from datetime import datetime, timezone, date
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from agent.config import config
from agent.database import Database
from agent.data_pipeline import DataPipeline
from agent.executor import Executor
from agent.signals.ema import compute_ema_state, drop_unclosed_daily_bar, _ema
from agent.signals import indicators as ind
from agent.signals import filters as filt
from agent.logging_setup import setup_logging
from agent import strategy
from agent import exits
from agent.exits import PositionState

# ── Component init ───────────────────────────────────────────────────
db = Database(config.database_url)
pipeline = DataPipeline()
executor = Executor()
NY_TZ = pytz.timezone("America/New_York")
STRATEGY_VERSION = "trendbuyer-v1"
_scheduler = None


# ── Helpers ──────────────────────────────────────────────────────────

def _dte_remaining(expiration: str) -> int:
    try:
        exp = datetime.strptime(expiration, "%Y-%m-%d").date()
        return max(0, (exp - date.today()).days)
    except Exception:
        return 999


def _current_side(symbol: str):
    t = db.get_open_trade_by_symbol(symbol)
    return t.option_type if t else None


def _option_price(trade) -> float:
    """Current per-share option price from the live position (fallback: entry price)."""
    pos = executor.get_position(trade.contract_symbol)
    if pos and pos.get("current_price"):
        return float(pos["current_price"])
    return float(trade.entry_price or 0.0)


def _option_gain_pct(trade) -> float:
    """Current option gain as a fraction of entry (e.g. +0.85). 0 if unknown."""
    entry = float(trade.entry_price or 0.0)
    if entry <= 0:
        return 0.0
    return (_option_price(trade) - entry) / entry


def _underlying_close_and_atr(symbol: str, now_et: datetime):
    """Latest CLOSED daily close, the 21-EMA, and a fresh chandelier-period ATR."""
    df = pipeline.get_daily_bars(symbol, period_days=config.strategy.history_days)
    df = drop_unclosed_daily_bar(df, now_et)
    if df is None or df.empty:
        return None, None, None
    close = float(df["close"].iloc[-1])
    chand_atr = ind.atr(df, config.exits.chandelier_atr_period)
    ema21 = float(_ema(df["close"].astype(float), config.strategy.ema_slow).iloc[-1])
    return close, chand_atr, ema21


# ── Closing / P&L bookkeeping ────────────────────────────────────────

def _book_close(trade, reason: str, exit_opt_px: float, contracts_closed: int):
    """Record a FULL close: total P&L = banked partials + (final proceeds − cost of the remainder)."""
    cost_remaining = float(trade.entry_price or 0.0) * 100 * contracts_closed
    final_proceeds = exit_opt_px * 100 * contracts_closed
    total = float(trade.realized_partial_pnl or 0.0) + (final_proceeds - cost_remaining)
    total_pct = total / trade.premium_paid if trade.premium_paid else 0.0
    db.update_trade(
        trade.id, status="closed", exit_time=datetime.now(timezone.utc),
        exit_reason=reason, exit_price=round(exit_opt_px, 2),
        pnl=round(total, 2), pnl_pct=round(total_pct, 4),
    )
    emoji = "🟢" if total > 0 else "🔴"
    logger.info(f"{emoji} EXIT {trade.symbol} ({trade.option_type}): {reason} | "
                f"total P&L=${total:+.0f} ({total_pct:+.0%})")


def _close_full(trade, reason: str) -> bool:
    """Close the whole remaining position and book it. Returns True on success."""
    opt_px = _option_price(trade)
    order = executor.close_option(trade.contract_symbol, reason=reason)
    if not order:
        logger.error(f"{trade.symbol}: close failed ({reason}) — will retry next cycle")
        return False
    _book_close(trade, reason, opt_px, int(trade.contracts or 1))
    return True


def _take_partial_and_arm(trade, close: float, peak: float):
    """Bank a partial scale-out (if we hold ≥2 contracts), then arm the Chandelier trail.
    With a single contract we can't sell half, so we arm the trail + breakeven floor instead
    (lock in 'can't lose past breakeven', ride the rest)."""
    n0 = int(trade.contracts_initial or trade.contracts or 1)
    cur = int(trade.contracts or n0)
    partial_qty = int(n0 * config.exits.partial_scale_fraction)
    opt_px = _option_price(trade)

    if partial_qty >= 1 and cur - partial_qty >= 1:
        order = executor.close_option(trade.contract_symbol, contracts=partial_qty, reason="partial_take")
        if order:
            proceeds = opt_px * 100 * partial_qty
            cost = float(trade.entry_price or 0.0) * 100 * partial_qty
            realized = float(trade.realized_partial_pnl or 0.0) + (proceeds - cost)
            db.update_trade(trade.id, contracts=cur - partial_qty, partial_done=True,
                            trail_armed=True, peak_close=peak, trail_streak=0,
                            realized_partial_pnl=round(realized, 2))
            logger.info(f"💰 PARTIAL {trade.symbol}: sold {partial_qty}/{n0} @ ${opt_px:.2f} "
                        f"(banked ${proceeds - cost:+.0f}); trailing the rest")
            return
    # Single contract (or can't scale) → arm trail + breakeven, no sale.
    db.update_trade(trade.id, partial_done=True, trail_armed=True, peak_close=peak, trail_streak=0)
    logger.info(f"🎯 {trade.symbol}: +{config.exits.partial_profit_pct:.0%} target hit on 1 contract — "
                f"trail armed at breakeven (no partial sale)")


# ── Daily-close monitor: time-exit → partial → hard stop → trail ─────

def run_close_monitor():
    now_et = datetime.now(NY_TZ)
    logger.info(f"═══ Close Monitor: {now_et.strftime('%Y-%m-%d %H:%M ET')} ═══")
    if not executor.is_market_open():
        logger.info("Market closed — skipping monitor")
        return

    for trade in db.get_open_trades():
        try:
            # One shared, pure decision (identical logic live and in backtest).
            dte = _dte_remaining(trade.expiration)
            close, chand_atr, ema21 = _underlying_close_and_atr(trade.symbol, now_et)
            chand_atr = chand_atr or trade.atr_at_entry or 0.0
            gain = _option_gain_pct(trade)
            dec = exits.decide_exit(PositionState.from_trade(trade), close, chand_atr,
                                    ema21, gain, dte, config)

            if dec.action == exits.TIME_EXIT:
                _close_full(trade, dec.reason)
            elif dec.action == exits.PARTIAL_ARM:
                _take_partial_and_arm(trade, close, dec.new_peak)
            elif dec.action in (exits.HARD_STOP, exits.TRAIL_EXIT):
                _close_full(trade, dec.reason)
            elif dec.action == exits.WAIT_STOP:
                db.update_trade(trade.id, stop_streak=dec.new_stop_streak, peak_close=dec.new_peak)
                logger.info(f"{trade.symbol}: {dec.reason} "
                            f"(close {close:.2f} vs stop {trade.hard_stop_price:.2f}) — waiting")
            elif dec.action == exits.WAIT_TRAIL:
                db.update_trade(trade.id, trail_streak=dec.new_trail_streak, peak_close=dec.new_peak)
                logger.info(f"{trade.symbol}: {dec.reason} "
                            f"(close {close:.2f} vs line {dec.trail_line:.2f}) — waiting")
            else:  # HOLD — persist refreshed peak + reset streaks (no-op when close is None)
                db.update_trade(trade.id, stop_streak=dec.new_stop_streak,
                                trail_streak=dec.new_trail_streak, peak_close=dec.new_peak)
        except Exception as e:
            logger.error(f"Monitor failed for {trade.contract_symbol}: {e}")

    snapshot_portfolio()
    logger.info("═══ Monitor done ═══\n")


# ── Portfolio guardrails for entries ─────────────────────────────────

def _portfolio_caps(open_trades):
    """Current open-book counts used to gate new entries."""
    symbols = {t.symbol for t in open_trades}
    calls = sum(1 for t in open_trades if t.option_type == "CALL")
    puts = sum(1 for t in open_trades if t.option_type == "PUT")
    by_sector = {}
    for t in open_trades:
        sec = config.account.sector_of(t.symbol)
        by_sector[sec] = by_sector.get(sec, 0) + 1
    return symbols, calls, puts, by_sector


def _direction_ok(option_type, calls, puts) -> bool:
    """Keep same-direction exposure under the cap (the anti-correlated-wipeout guard)."""
    allowed_same = max(1, round(config.account.max_positions * config.account.direction_cap_pct))
    same = (calls if option_type == "CALL" else puts) + 1
    return same <= allowed_same


# ── Entry routine ────────────────────────────────────────────────────

def run_open_routine():
    now_et = datetime.now(NY_TZ)
    logger.info(f"═══ Open Routine: {now_et.strftime('%Y-%m-%d %H:%M ET')} ═══")
    if not executor.is_market_open():
        logger.info("Market closed (holiday?) — skipping")
        return
    account = executor.get_account()
    if account.get("trading_blocked"):
        logger.error("Account trading blocked — halting")
        return
    equity, cash = account["equity"], account["cash"]
    open_trades = db.get_open_trades()
    symbols, calls, puts, by_sector = _portfolio_caps(open_trades)

    for symbol in config.strategy.watchlist:
        ema = None
        action, reason, executed = "SKIP", "no data", False
        metrics = {}
        try:
            df = pipeline.get_daily_bars(symbol, period_days=config.strategy.history_days)
            df = drop_unclosed_daily_bar(df, now_et)
            ema = compute_ema_state(symbol, df)
            if ema is None:
                continue

            decision = strategy.decide(ema, _current_side(symbol))
            action, reason = decision.action, decision.reason
            logger.info(f"{symbol}: close=${ema.close:.2f} EMA9={ema.ema_fast:.2f} EMA21={ema.ema_slow:.2f} "
                        f"cross={ema.cross} age={ema.cross_age_days} fresh={ema.fresh} -> {action}")
            if decision.action not in ("BUY_CALL", "BUY_PUT"):
                continue

            direction = "bull" if decision.option_type == "CALL" else "bear"

            # --- Filters: turn the raw cross into a confirmed-trend signal ---
            fr = filt.evaluate(df, direction)
            metrics = fr.metrics
            if not fr.passed:
                action, reason = "SKIP", f"filtered: {fr.reason}"
                logger.info(f"{symbol}: SKIP — {fr.reason}")
                continue

            # --- Portfolio caps ---
            existing = db.get_open_trade_by_symbol(symbol)
            if not existing and len(symbols) >= config.account.max_positions:
                action, reason = "SKIP", f"max positions {config.account.max_positions} reached"
                logger.info(f"{symbol}: SKIP — {reason}")
                continue
            if not existing and not _direction_ok(decision.option_type, calls, puts):
                action, reason = "SKIP", f"direction cap (too many {decision.option_type}s)"
                logger.info(f"{symbol}: SKIP — {reason}")
                continue
            sec = config.account.sector_of(symbol)
            if not existing and by_sector.get(sec, 0) >= config.account.max_per_sector:
                action, reason = "SKIP", f"sector cap ({sec} full)"
                logger.info(f"{symbol}: SKIP — {reason}")
                continue

            # --- Flip: close the opposite side first ---
            if existing:
                if _close_full(existing, "ema_flip"):
                    cash = executor.get_account()["cash"]
                    open_trades = db.get_open_trades()
                    symbols, calls, puts, by_sector = _portfolio_caps(open_trades)

            # --- Pick the deep-ITM contract (deeper when vol is expensive) ---
            tdelta = strategy.target_delta_for(metrics.get("vol_rank"))
            contract = executor.find_option_contract(symbol, decision.option_type, target_delta=tdelta)
            if not contract:
                action, reason = "SKIP", f"{decision.reason} — no deep-ITM contract"
                logger.info(f"{symbol}: no deep-ITM {decision.option_type} contract — skipping")
                continue

            # --- Size on the STOP distance (aggressive risk, hard caps) ---
            atr_val = metrics.get("atr") or ind.atr(df, config.filters.atr_period) or 0.0
            delta = abs(contract.get("delta") or tdelta)
            ask_per_contract = max(float(contract.get("ask") or 0) * 100, contract["premium_per_contract"])
            est_loss = strategy.estimate_stop_loss_per_contract(delta, atr_val, config.exits.hard_stop_atr_mult)
            sized = strategy.size_position(equity, cash, ask_per_contract, est_loss)
            if sized.contracts < 1:
                action, reason = "SKIP", f"sizing: {sized.reason}"
                logger.info(f"{symbol}: SKIP — sizing rejected ({sized.reason})")
                continue

            # --- Buy (limit walked from mid; honest fills) ---
            order = executor.buy_option(contract, contracts=sized.contracts, reason=decision.reason)
            if not order:
                action, reason = "SKIP", f"{decision.reason} — order didn't fill fairly"
                continue

            filled = int(order.get("contracts", sized.contracts))
            cash -= order["total_cost"]
            executed = True
            entry_px = float(metrics.get("price") or ema.close)
            hard_stop = strategy.hard_stop_level(entry_px, atr_val, decision.option_type)

            db.create_trade(
                symbol=symbol, side="buy",
                contract_symbol=order["occ_symbol"], option_type=order["option_type"],
                strategy_type=decision.action, strike=order["strike"], expiration=order["expiration"],
                dte_at_entry=order["dte"], iv_at_entry=order.get("implied_volatility"),
                contracts=filled, contracts_initial=filled,
                entry_price=order["premium_per_contract"] / 100, premium_paid=order["total_cost"],
                entry_time=datetime.now(timezone.utc), status="open",
                delta_at_entry=delta, vol_rank_at_entry=metrics.get("vol_rank"),
                entry_underlying=round(entry_px, 2), atr_at_entry=round(atr_val, 2),
                hard_stop_price=round(hard_stop, 2), peak_close=round(entry_px, 2),
                alpaca_order_id=order.get("order_id"), strategy_version=STRATEGY_VERSION,
            )
            # Update running caps so later symbols see this position.
            symbols.add(symbol)
            if decision.option_type == "CALL":
                calls += 1
            else:
                puts += 1
            by_sector[sec] = by_sector.get(sec, 0) + 1
            logger.success(f"✅ {decision.action} {symbol}: {filled}x deep-ITM @ "
                           f"${order['premium_per_contract']/100:.2f}/sh | cost=${order['total_cost']:.0f} | "
                           f"stop@{hard_stop:.2f} (underlying) | risk≈${est_loss*filled:.0f}")
        except Exception as e:
            action, reason = "SKIP", f"error: {e}"
            logger.error(f"{symbol}: entry error: {e}")
        finally:
            db.log_signal(
                symbol=symbol,
                close=ema.close if ema else None,
                ema_fast=ema.ema_fast if ema else None,
                ema_slow=ema.ema_slow if ema else None,
                fast_above_slow=ema.fast_above_slow if ema else None,
                cross=ema.cross if ema else None,
                cross_age_days=ema.cross_age_days if ema else None,
                fresh=ema.fresh if ema else False,
                action=action, reason=reason, executed=executed, metrics_json=metrics or None,
            )

    snapshot_portfolio()
    logger.info("═══ Open routine done ═══\n")


# ── Snapshot ─────────────────────────────────────────────────────────

def snapshot_portfolio():
    if not executor.is_market_open():
        return
    try:
        account = executor.get_account()
        positions = executor.get_positions()
        equity = account["equity"]
        invested = sum(p.get("market_value", 0) for p in positions)
        num_opt = sum(1 for p in positions if p.get("asset_class") == "us_option")
        daily_pnl = equity - account["last_equity"]
        start_of_day = equity - daily_pnl
        total_pnl = equity - config.account.initial_capital
        db.log_portfolio_snapshot(
            equity=equity, cash=account["cash"], invested_value=invested, num_positions=num_opt,
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_pct=round(daily_pnl / start_of_day, 4) if start_of_day else 0,
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl / config.account.initial_capital, 4) if config.account.initial_capital else 0,
            positions_json=positions,
        )
        logger.info(f"💓 snapshot: equity=${equity:,.0f} cash=${account['cash']:,.0f} "
                    f"positions={num_opt} day P&L=${daily_pnl:+,.0f}")
    except Exception as e:
        logger.error(f"Snapshot failed: {e}")


# ── Scheduling / lifecycle ───────────────────────────────────────────

def register_jobs(sched):
    sched.add_job(run_open_routine,
                  CronTrigger(day_of_week="mon-fri", hour=config.scheduler.eval_hour,
                              minute=config.scheduler.eval_minute, timezone=NY_TZ),
                  misfire_grace_time=2 * 3600, coalesce=True)
    sched.add_job(run_close_monitor,
                  CronTrigger(day_of_week="mon-fri", hour=config.scheduler.monitor_hour,
                              minute=config.scheduler.monitor_minute, timezone=NY_TZ),
                  misfire_grace_time=20 * 60, coalesce=True)
    sched.add_job(snapshot_portfolio,
                  CronTrigger(day_of_week="mon-fri", hour="9-16",
                              minute=f"*/{config.scheduler.snapshot_interval_minutes}", timezone=NY_TZ),
                  misfire_grace_time=300, coalesce=True)


def _log_startup():
    mode = "📄 PAPER" if config.alpaca.paper else "💰 LIVE"
    logger.info(
        f"Mode: {mode} | Capital: ${config.account.initial_capital:,.0f} | "
        f"Watchlist: {len(config.strategy.watchlist)} | EMA {config.strategy.ema_fast}/{config.strategy.ema_slow} daily | "
        f"deep-ITM Δ{config.options.target_delta:.2f} ~{config.options.target_dte}DTE | "
        f"risk {config.account.risk_per_trade_pct:.0%}/trade, max {config.account.max_positions} positions"
    )


def shutdown(signum, frame):
    logger.info("Shutting down...")
    if _scheduler:
        _scheduler.shutdown(wait=False)
    sys.exit(0)


def run_background():
    global _scheduler
    setup_logging()
    if not config.validate():
        logger.warning("Config invalid (check API keys) — bot will idle until fixed.")
    _scheduler = BackgroundScheduler(timezone=NY_TZ)
    register_jobs(_scheduler)
    _log_startup()
    try:
        snapshot_portfolio()
    except Exception:
        pass
    _scheduler.start()
    logger.info("Scheduler started (background).")
    return _scheduler


def run():
    global _scheduler
    setup_logging()
    logger.info("🚀 Trend-Buyer v1 starting...")
    if not config.validate():
        logger.error("Fix config errors above before starting.")
        sys.exit(1)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    _scheduler = BlockingScheduler(timezone=NY_TZ)
    register_jobs(_scheduler)
    _log_startup()
    try:
        snapshot_portfolio()
    except Exception:
        pass
    logger.info("Running. Ctrl+C to stop.")
    _scheduler.start()


if __name__ == "__main__":
    run()
