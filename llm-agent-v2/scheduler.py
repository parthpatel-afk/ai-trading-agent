"""
AI Investor Agent - Main Agent Loop v2.0 (Options Trading)
Entry point: python scheduler.py

Key fixes vs v1.0:
  - Real daily P&L (no more hardcoded 0.0)
  - Risk settings loaded from DB each cycle (dashboard changes propagate live)
  - Options-specific exit checks (50% stop, 7 DTE, profit target)
  - Peak value tracked in DB for correct exit calculations
"""
import re
import signal
import sys
import threading
from datetime import datetime, timezone, date
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from agent.config import config
from agent.database import Database
from agent.data_pipeline import DataPipeline
from agent.feedback.tuner import ParameterTuner
from agent.signals.technical import compute_technical_signals, check_hourly_macd
from agent.signals.sentiment import score_sentiment
from agent.signals.macro import compute_macro_signals
from agent.signals.options_flow import compute_options_flow
from agent.signals.reversal import compute_reversal_signals
from agent.signals.pre_screener import pre_screen
from agent.brain import Brain, ENTRY_ACTIONS
from agent.risk import RiskEngine, ExitSignal
from agent.executor import Executor

# ── Logging ──────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add("logs/agent_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")

# ── Component init ───────────────────────────────────────────────────
db = Database(config.database_url)
db.ensure_risk_settings_exist()    # Seed DB with config defaults if first run
pipeline = DataPipeline()
brain = Brain()
risk = RiskEngine()
executor = Executor()
tuner = ParameterTuner(db)
NY_TZ = pytz.timezone("America/New_York")
scheduler = BlockingScheduler(timezone=NY_TZ)
STRATEGY_VERSION = "v2.0"

# Symbols closed today — prevents re-entering same position same day (avoids PDT accumulation)
_closed_today: set = set()

# Concurrency guard: run_agent_cycle (15-min) and monitor_positions (2-min) run as
# separate APScheduler threads and can fire on the same second. Without this, both
# threads pass the "is it still open?" DB check and submit a sell for the same contract —
# the second sell hits a position that's already being closed and Alpaca rejects it as
# "uncovered" (seen on ARKK 2026-06-05). This lets only one thread close a given contract.
_exit_lock = threading.Lock()
_closing_contracts: set = set()

# Smart stop-loss state (2026-06-07). Keyed by contract OCC symbol.
#  _stock_exit_streak: consecutive cycles an immediate stock-based exit (hard thesis
#    break / chandelier) has been true — must reach thesis_break_confirm_cycles to fire.
#  _gutchecked_contracts: positions that already had their one-off Layer-3 AI gut-check
#    (so we spend at most one GPT call per losing position, not one per cycle).
_stock_exit_streak: dict = {}
_gutchecked_contracts: set = set()

#  _credit_stop_streak: consecutive cycles a credit-spread STOP (cost-to-close ≥ 2× credit)
#    has read over the line — must reach credit_stop_confirm_cycles before firing. Stops also
#    suppressed in the first stop_warmup_minutes after the open. Both added 2026-06-12 after
#    NVDA stopped on a junk opening mark ($215 cost-to-close that was really ~$105).
_credit_stop_streak: dict = {}

# Anti-chase guard (2026-06-07): the first price at which each symbol entered the
# candidate pool today — {symbol: (price, date)}. Used to measure how far a name has
# run before we actually buy, so we don't chase an ATM option at the highs. Reset daily.
_candidate_first_seen: dict = {}


def run_agent_cycle():
    """Main scan: Data → Signals → Brain → Risk → Execute → Log"""
    now = datetime.now(pytz.timezone("America/New_York"))
    logger.info(f"═══ Agent Cycle: {now.strftime('%H:%M ET')} ═══")

    if not executor.is_market_open():
        logger.info("Market closed — skipping")
        return

    account = executor.get_account()
    if account.get("trading_blocked"):
        logger.error("Account trading blocked. Halting cycle.")
        return

    equity = account["equity"]
    cash = account["cash"]

    # ── Sync risk settings from DB (cross-process fix) ───────────────
    risk.sync_settings_from_db(db)

    # ── Portfolio health check ────────────────────────────────────────
    positions = executor.get_positions()
    num_positions = _open_bet_count(positions)

    # ── Check exits on open positions — ALWAYS runs, even when unhealthy ─────
    # Exit monitoring must not be blocked by the daily loss limit gate.
    # Positions need to be watched every cycle regardless of portfolio health.
    open_trades = db.get_open_trades()
    open_contracts = _known_spread_contracts(open_trades)
    for trade in open_trades:
        try:
            _check_option_exit(trade)
        except Exception as e:
            logger.error(f"Exit check failed for {trade.contract_symbol}: {e}")

    # ── Orphan check: Alpaca option positions not tracked in DB ───────
    for pos in positions:
        if "option" in pos.get("asset_class", "").lower() and pos["symbol"] not in open_contracts:
            try:
                _close_orphan_if_needed(pos)
            except Exception as e:
                logger.error(f"Orphan check failed for {pos['symbol']}: {e}")

    # ── Daily P&L = mark-to-market change since the prior session close
    #    (Alpaca's last_equity), NOT the lifetime sum of trades that closed today.
    #    A loser that bled over 5 days but closed today should only count TODAY's
    #    slice against the daily limit, not its entire lifetime loss. Because an
    #    open position's unrealized loss is already in `equity`, this also gates
    #    new entries correctly whether or not a same-cycle exit has fired yet. ──
    daily_pnl = equity - account["last_equity"]

    healthy, msg = risk.check_portfolio_health(equity, daily_pnl, num_positions, db=db)
    _log_portfolio(equity, cash, positions, daily_pnl, circuit_breaker=not healthy)

    if not healthy:
        logger.warning(msg)
        return

    # ── Scan for new entries: 3-phase batch flow ──────────────────────
    macro = compute_macro_signals()
    # Tie the number of concurrent bets to market mood: fewer open positions when the
    # regime is weak/shaky (scaled by regime_score, hard floor of 1). The bot already
    # shrinks bet *size* in bad regimes; this also throttles how *many* bets it carries
    # so it can't pile from 2→6 positions in a falling market (as it did 2026-06-04).
    effective_max = max(1, int(config.risk.max_positions * macro.regime_score + 0.5))
    if effective_max < config.risk.max_positions:
        logger.info(
            f"Regime {macro.regime}: position cap trimmed to "
            f"{effective_max}/{config.risk.max_positions} open bets"
        )
    if num_positions < effective_max:
        open_underlyings = {t.symbol for t in open_trades if t.status == "open"}
        # Also block any underlying that has a live Alpaca option position but is missing
        # from the DB (e.g. orphan positions created by duplicate orders). This prevents
        # the bot from re-entering a stock it already has exposure to.
        for pos in positions:
            if "option" in pos.get("asset_class", "").lower():
                m = re.match(r'^([A-Z]+)\d', pos.get("symbol", ""))
                if m:
                    open_underlyings.add(m.group(1))
        available_slots = effective_max - num_positions

        # Phase 1: Cheap technical scan — all eligible symbols, no API cost.
        # With the credit engine on, scan the tightened, liquid CORE universe (spreads
        # need tight option markets); otherwise use the full directional watchlist.
        active_watchlist = (
            config.strategy.credit_core_universe
            if config.options.enable_credit_spreads
            else config.strategy.watchlist
        )
        candidates = []
        for symbol in active_watchlist:
            if symbol in open_underlyings:
                continue
            try:
                res = _quick_screen(symbol)
                if res:
                    candidates.append((symbol, res[0], res[1]))  # (symbol, technical, direction)
            except Exception as e:
                logger.error(f"{symbol}: Quick screen error: {e}")

        if not candidates:
            logger.info("No candidates passed technical screen this cycle")
        else:
            # Strongest setups first — rank each by the score matching its direction.
            candidates.sort(
                key=lambda x: (x[1].bearish_technical_score if x[2] == "down" else x[1].technical_score),
                reverse=True,
            )
            logger.info(f"{len(candidates)} technical candidates: {[(s, d) for s, _, d in candidates]}")

            # Phase 2: Full pipeline on all candidates (Groq → sentiment → options → Gemini)
            bundles = []
            for symbol, technical, direction in candidates:
                try:
                    bundle = _full_pipeline(symbol, macro, technical, direction, equity)
                    if bundle:
                        bundles.append(bundle)
                except Exception as e:
                    logger.error(f"{symbol}: Full pipeline error: {e}")

            if not bundles:
                logger.info("No candidates survived full pipeline")
            else:
                logger.info(f"{len(bundles)} pipeline survivors for batch ranking: {[b['symbol'] for b in bundles]}")

                # Phase 3: One GPT-4o call ranks all candidates head-to-head
                ranked = brain.decide_batch(bundles, available_slots)

                for decision, bundle in ranked:
                    db.log_decision(
                        symbol=decision.symbol,
                        action=decision.action,
                        confidence=decision.confidence,
                        reasoning=decision.reasoning,
                        signal_summary=decision.score_breakdown,
                        strategy_mode=decision.strategy_mode,
                        target_dte=decision.target_dte,
                        suggested_pct=decision.position_size_pct,
                        executed=False,
                        execution_blocked_reason=decision.blocked_reason or None,
                        strategy_version=STRATEGY_VERSION,
                    )
                    tech = bundle["technical"]
                    sent = bundle["sentiment"]
                    opts = bundle["options_flow"]
                    db.log_signal_snapshot(
                        symbol=decision.symbol,
                        rsi=tech.rsi,
                        macd=tech.macd,
                        macd_signal=tech.macd_signal_line,
                        macd_hist=tech.macd_hist,
                        macd_crossover=tech.macd_crossover,
                        ema_9=tech.ema_9,
                        ema_21=tech.ema_21,
                        ema_50=tech.ema_50,
                        above_ema_trend=tech.above_ema_50,
                        bollinger_pct=tech.bollinger_pct,
                        atr=tech.atr,
                        volume_ratio=tech.volume_ratio,
                        sentiment_score=sent.overall_score,
                        news_count=sent.news_count,
                        sentiment_summary=sent.summary,
                        sentiment_from_cache=sent.from_cache,
                        vix_level=bundle["macro"].vix_level,
                        market_regime=bundle["macro"].regime,
                        iv_rank=opts.iv_rank,
                        iv_regime=opts.iv_regime,
                        put_call_ratio=opts.put_call_ratio,
                        earnings_blackout=opts.earnings_within_blackout,
                        strategy_mode=opts.strategy_mode,
                        close_price=tech.close,
                        volume=tech.volume,
                        technical_score=tech.technical_score,
                        options_flow_score=decision.options_flow_score,
                        overall_confidence=decision.composite_score,
                    )

                    if decision.action not in ENTRY_ACTIONS:
                        continue
                    if num_positions >= effective_max:
                        break
                    try:
                        entered = _execute_decision(
                            decision, account, bundle.get("contract"), bundle.get("spread")
                        )
                        if entered:
                            num_positions += 1
                    except Exception as e:
                        logger.error(f"{decision.symbol}: Execution error: {e}")

    logger.info("═══ Cycle done ═══\n")


def _known_spread_contracts(open_trades) -> set:
    """
    Every option contract the DB knows about — BOTH legs of each open spread.
    A credit spread is stored as one row keyed on the SHORT leg (contract_symbol),
    with the protective long leg in long_leg_symbol. The orphan check must treat the
    long leg as known too; otherwise it flags the insurance leg as an "orphan" every
    cycle and could close it on its own, leaving a naked short with uncapped risk.
    """
    known = set()
    for t in open_trades:
        if t.status != "open":
            continue
        if t.contract_symbol:
            known.add(t.contract_symbol)
        if getattr(t, "long_leg_symbol", None):
            known.add(t.long_leg_symbol)
    return known


def _open_bet_count(positions) -> int:
    """
    Count BETS, not legs. A credit spread shows up as 2 Alpaca option legs (short + long)
    on the same underlying but is ONE bet. Counting raw legs made the bot read 2 spreads
    as 4 positions, tripping the position cap and blocking new entries. Group option legs
    by underlying ticker so each spread counts once; count any non-option position singly.
    """
    underlyings = set()
    non_option = 0
    for p in positions:
        if "option" in p.get("asset_class", "").lower():
            m = re.match(r'^([A-Z]+)\d', p.get("symbol", ""))
            underlyings.add(m.group(1) if m else p.get("symbol", ""))
        else:
            non_option += 1
    return len(underlyings) + non_option


def _close_orphan_if_needed(pos: dict):
    """
    Handle an Alpaca option position that has no matching DB open trade.
    Reconstructs trade params from the OCC symbol + Alpaca position data,
    then applies full exit logic. Closes the position if any trigger fires.
    """
    occ = pos["symbol"]
    m = re.match(r'^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$', occ)
    if not m:
        logger.warning(f"Orphan: cannot parse OCC symbol {occ} — skipping")
        return

    underlying = m.group(1)
    option_type = "call" if m.group(5) == "C" else "put"
    strike = int(m.group(6)) / 1000.0

    from datetime import date
    exp_date = date(int(f"20{m.group(2)}"), int(m.group(3)), int(m.group(4)))
    dte_remaining = max(0, (exp_date - date.today()).days)

    contracts = int(pos["qty"])
    premium_paid = pos["avg_entry_price"] * 100 * contracts
    current_value = pos["current_price"] * 100 * contracts

    try:
        stock_price = pipeline.get_current_price(underlying) or 0.0
    except Exception:
        stock_price = 0.0

    logger.warning(
        f"Orphan position {occ}: underlying={underlying} strike={strike} "
        f"dte={dte_remaining} current=${current_value:.0f} paid=${premium_paid:.0f}"
    )

    exit_signal = risk.check_option_exit(
        symbol=underlying,
        premium_paid=premium_paid,
        current_value=current_value,
        dte_remaining=dte_remaining,
        peak_value=premium_paid,  # peak unknown — use premium_paid (conservative)
        stock_price=stock_price,
        strike=strike,
        iv=0.0,          # IV unknown — skips delta floor check
        option_type=option_type,
        days_held=0,     # days held unknown — skips velocity check
        contracts=contracts,
    )

    if exit_signal.should_exit:
        logger.info(f"Orphan {occ}: closing — {exit_signal.reason}")
        order = executor.close_option(occ, reason=f"orphan: {exit_signal.reason}")
        if order:
            pnl = current_value - premium_paid
            logger.info(f"Orphan {occ} closed. Estimated P&L: ${pnl:+.0f}")
    else:
        logger.info(f"Orphan {occ}: holding — {exit_signal.reason}")


def _find_spread_short_leg(underlying: str, long_occ: str) -> str:
    """Find the short leg OCC for an open spread by scanning current positions."""
    positions = executor.get_positions()
    for p in positions:
        sym = p.get("symbol", "")
        side = p.get("side", "")
        asset_class = p.get("asset_class", "")
        if "option" in asset_class.lower() and "short" in side.lower() and sym.startswith(underlying) and sym != long_occ:
            return sym
    return None


def _quick_screen(symbol: str):
    """
    Phase 1: Technical-only screen. No API cost. Returns (TechnicalSignals, direction) or None.
    direction ∈ {"up","down"}. When the credit engine is on the screen is TWO-SIDED (also
    hunts downtrends → puts / bear call spreads); otherwise it is bullish-only (legacy).
    """
    if symbol in _closed_today:
        logger.info(f"{symbol}: Skip — closed today (cooldown)")
        return None

    df = pipeline.get_historical_bars(symbol, period_days=90)
    if df is None or len(df) < 55:
        logger.info(f"{symbol}: Skip — insufficient data ({len(df) if df is not None else 0} bars)")
        return None

    technical = compute_technical_signals(symbol, df)
    if technical is None:
        logger.info(f"{symbol}: Skip — technical compute failed")
        return None

    two_sided = config.options.enable_credit_spreads
    hourly_df = pipeline.get_hourly_bars(symbol)
    h_bullish, h_macd, h_sig = check_hourly_macd(hourly_df)
    h_bearish = h_macd < h_sig  # fails open to bullish (h_macd=h_sig=0 → not bearish), safe for shorts

    bullish_ok = technical.entry_signal and h_bullish
    bearish_ok = two_sided and technical.bearish_entry_signal and h_bearish

    if not bullish_ok and not bearish_ok:
        rsi_str = f"RSI={technical.rsi:.0f}" if technical.rsi else "RSI=n/a"
        reason = (
            f"no setup — {rsi_str}, above_EMA50={technical.above_ema_50}, "
            f"hourly MACD {'bull' if h_bullish else 'bear'} ({h_macd:.3f} vs {h_sig:.3f}), "
            f"score↑{technical.technical_score:.2f}/↓{technical.bearish_technical_score:.2f}"
        )
        logger.info(f"{symbol}: Skip (technical) — {reason}")
        db.log_decision(
            symbol=symbol, action="SKIP", confidence=0.0,
            reasoning=f"No technical setup: {reason}",
            signal_summary={
                "technical_score": technical.technical_score,
                "bearish_technical_score": technical.bearish_technical_score,
                "rsi": technical.rsi, "above_ema_50": technical.above_ema_50,
                "h_macd": h_macd, "h_signal": h_sig,
            },
            strategy_version=STRATEGY_VERSION,
        )
        return None

    # If both sides somehow qualify, take the stronger one.
    if bullish_ok and bearish_ok:
        direction = "up" if technical.technical_score >= technical.bearish_technical_score else "down"
    else:
        direction = "up" if bullish_ok else "down"

    # Record the first price this symbol appeared as a candidate today (anti-chase guard).
    today = date.today()
    prev = _candidate_first_seen.get(symbol)
    if prev is None or prev[1] != today:
        _candidate_first_seen[symbol] = (technical.close, today)

    logger.info(f"{symbol}: candidate ({direction})")
    return technical, direction


def _full_pipeline(symbol: str, macro, technical, direction: str = "up", equity: float = 10000.0):
    """
    Phase 2: Full signal pipeline for a candidate that passed Phase 1.
    Runs Groq → sentiment → options flow → Gemini. Returns bundle dict or None if rejected.
    `direction` ("up"/"down") routes the strategy and picks the directional strength score.
    """
    # The Groq pre-screener is bullish-biased — only gate "up" candidates with it; let
    # "down" candidates through (they were validated by the bearish technical screen).
    if direction == "up":
        screen = pre_screen(symbol, technical)
        if not screen.worth_analyzing:
            logger.info(f"{symbol}: Skip (pre-screener) — {screen.reason}")
            db.log_decision(
                symbol=symbol, action="SKIP", confidence=0.0,
                reasoning=f"Pre-screener: {screen.reason}",
                signal_summary={"technical_score": technical.technical_score},
                strategy_version=STRATEGY_VERSION,
            )
            return None

    news = pipeline.get_company_news(symbol, hours_back=config.strategy.news_lookback_hours)
    sentiment = score_sentiment(symbol, news)

    # Strength score that matches the candidate's direction (bullish vs bearish scorer).
    directional_score = technical.technical_score if direction == "up" else technical.bearish_technical_score
    options_flow = compute_options_flow(
        symbol=symbol,
        technical_score=directional_score,
        sentiment_score=sentiment.overall_score,
        direction=direction,
    )

    if options_flow.earnings_within_blackout:
        logger.info(f"{symbol}: Skipped — earnings blackout ({options_flow.earnings_date})")
        return None

    if options_flow.strategy_mode == "SKIP":
        logger.info(
            f"{symbol}: Skip (options_flow) — tech_score={technical.technical_score:.2f}, "
            f"sentiment={sentiment.overall_score:.2f}, IV_rank={options_flow.iv_rank}, "
            f"earnings={options_flow.earnings_within_blackout} | {options_flow.notes.strip()}"
        )
        return None

    quote = pipeline.get_latest_quote(symbol)
    price = quote["mid_price"] if quote and quote.get("mid_price") else technical.close

    # ── Credit-spread path (sell premium when IV is rich) ─────────────
    # When the strategy mode is a credit spread, build the defined-risk spread now (both
    # legs + credit/width economics) so the brain ranks it on credit quality, not breakeven.
    CREDIT_MODES = {"SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "IRON_CONDOR"}
    if options_flow.strategy_mode in CREDIT_MODES:
        if options_flow.strategy_mode == "IRON_CONDOR":
            spread = executor.find_iron_condor_contracts(symbol, target_dte=35)
        else:
            spread = executor.find_credit_spread_contracts(symbol, options_flow.strategy_mode, target_dte=35)
        if not spread:
            logger.info(f"{symbol}: Skip — no suitable {options_flow.strategy_mode} found")
            return None
        # Quality gate (mirror of risk.check_credit_spread_entry): worthwhile credit vs width.
        if spread["credit_to_width"] < config.options.min_credit_to_width:
            logger.info(
                f"{symbol}: Skip ({options_flow.strategy_mode}) — credit/width "
                f"{spread['credit_to_width']:.0%} < {config.options.min_credit_to_width:.0%} (poor risk/reward)"
            )
            return None
        # Affordability: one spread's max loss must fit the per-trade risk budget.
        budget = equity * config.options.max_loss_pct_per_trade
        if spread["max_loss_per_spread"] > budget:
            logger.info(
                f"{symbol}: Skip ({options_flow.strategy_mode}) — one spread risks "
                f"${spread['max_loss_per_spread']:.0f} > ${budget:.0f} per-trade budget"
            )
            return None
        logger.debug(
            f"{symbol}: Spread locked — {options_flow.strategy_mode} credit=${spread['credit_per_spread']:.0f} "
            f"width=${spread['width_per_spread']:.0f} max_loss=${spread['max_loss_per_spread']:.0f}"
        )
        return {
            "symbol": symbol,
            "technical": technical,
            "sentiment": sentiment,
            "macro": macro,
            "options_flow": options_flow,
            "current_price": price,
            "spread": spread,
            "direction": direction,
            "directional_score": directional_score,
        }

    # Find the actual contract NOW — before the brain call — so GPT-4o knows exactly
    # what it's deciding on: specific strike, cost, and how far the stock needs to move.
    option_type = "CALL" if options_flow.strategy_mode == "BUY_CALL" else "PUT"

    # ── Satellite reversal gate (Phase 4) ─────────────────────────────
    # When reversal timing is on, only BUY on a CONFIRMED reversal/exhaustion in the right
    # direction — buy the bounce at a tired bottom (calls) / the rejection at a tired top
    # (puts). Stops the "bought the top of the pop" failure (XLF/AAPL). Credit spreads are
    # NOT gated by this — they trade the trend+IV regime.
    itm_offset = 0.0
    if config.options.enable_reversal:
        df_rev = pipeline.get_historical_bars(symbol, period_days=90)
        rev = compute_reversal_signals(symbol, df_rev)
        want = rev.reversal_up if option_type == "CALL" else rev.reversal_down
        if not want:
            logger.info(f"{symbol}: Skip (no confirmed reversal for {option_type}) — "
                        f"{rev.notes.strip() or 'no exhaustion/divergence signal'}")
            db.log_decision(
                symbol=symbol, action="SKIP", confidence=0.0,
                reasoning=f"Satellite gate: no confirmed reversal {option_type} setup ({rev.notes.strip()}).",
                signal_summary={"reversal_up": rev.reversal_up, "reversal_down": rev.reversal_down,
                                "bearish_div": rev.bearish_divergence, "bullish_div": rev.bullish_divergence},
                strategy_version=STRATEGY_VERSION,
            )
            return None
        # Satellite buys are ITM (stock-like, low theta) and longer-dated — never the lottery ticket.
        itm_offset = config.options.satellite_itm_pct

    # ── Anti-chase / extension guard ──────────────────────────────────
    # Measure how far the stock has run (in daily-ATR multiples) since it first appeared
    # as a candidate today. Heavily extended → skip and wait for a pullback. Mildly
    # extended → don't buy ATM at the high; step to a slightly OTM, further-dated contract.
    target_dte = config.options.satellite_dte if config.options.enable_reversal else 35
    otm_offset = 0.0
    first = _candidate_first_seen.get(symbol)
    atr = technical.atr or 0.0
    if config.options.enable_chase_guard and first and first[1] == date.today() and atr > 0:
        run = (price - first[0]) if option_type == "CALL" else (first[0] - price)
        atr_run = run / atr
        if atr_run >= config.options.chase_atr_skip:
            logger.info(
                f"{symbol}: Skip (anti-chase) — ran {atr_run:.1f}×ATR from first-seen "
                f"${first[0]:.2f} to ${price:.2f}; waiting for a pullback instead of chasing"
            )
            db.log_decision(
                symbol=symbol, action="SKIP", confidence=0.0,
                reasoning=(f"Anti-chase: stock extended {atr_run:.1f}×ATR since first appearing "
                           f"as a candidate today (${first[0]:.2f}→${price:.2f}) — not chasing the high."),
                signal_summary={"atr_run": round(atr_run, 2), "first_seen": first[0],
                                "price": price, "atr": atr},
                strategy_version=STRATEGY_VERSION,
            )
            return None
        elif atr_run >= config.options.chase_atr_step and not config.options.enable_reversal:
            # Mild-extension OTM step only in legacy mode; satellite buys go ITM instead.
            otm_offset = config.options.chase_step_otm_pct
            target_dte = config.options.chase_step_dte
            logger.info(
                f"{symbol}: Extended {atr_run:.1f}×ATR from first-seen — stepping to "
                f"~{otm_offset:.0%} OTM / {target_dte} DTE instead of chasing ATM at the high"
            )

    contract = executor.find_option_contract(symbol, option_type, target_dte=target_dte,
                                             otm_offset_pct=otm_offset, itm_offset_pct=itm_offset)
    if not contract:
        logger.info(f"{symbol}: Skip — no suitable {option_type} contract found")
        return None

    max_premium = equity * config.risk.max_position_pct
    if contract["premium_per_contract"] > max_premium:
        logger.info(
            f"{symbol}: Skip — contract costs ${contract['premium_per_contract']:.0f}, "
            f"exceeds ${max_premium:.0f} max ({config.risk.max_position_pct:.0%} of ${equity:.0f} equity)"
        )
        return None

    # Compute trade mechanics the brain needs to reason about
    strike = contract["strike"]
    mid = contract["mid_price"]
    if option_type == "CALL":
        move_needed_pct = (strike - price) / price * 100      # positive = stock must rise
        breakeven = strike + mid
    else:
        move_needed_pct = (price - strike) / price * 100      # positive = stock must fall
        breakeven = strike - mid
    breakeven_move_pct = abs(breakeven - price) / price * 100

    contract["move_needed_pct"] = round(move_needed_pct, 1)
    contract["breakeven"] = round(breakeven, 2)
    contract["breakeven_move_pct"] = round(breakeven_move_pct, 1)

    logger.debug(
        f"{symbol}: Contract locked — {contract['occ_symbol']} | "
        f"move needed: {move_needed_pct:+.1f}% | breakeven: ${breakeven:.2f} ({breakeven_move_pct:.1f}% away)"
    )

    return {
        "symbol": symbol,
        "technical": technical,
        "sentiment": sentiment,
        "macro": macro,
        "options_flow": options_flow,
        "current_price": price,
        "contract": contract,
        "direction": direction,
        "directional_score": directional_score,
    }


def _execute_decision(decision, account: dict, contract: dict = None, spread: dict = None) -> bool:
    """Place the order for a batch-ranked entry decision. Returns True if filled.
    Routes credit-spread actions to the defined-risk sell path; everything else buys long."""
    if decision.action in {"SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "IRON_CONDOR"}:
        return _execute_credit_spread(decision, account, spread)

    symbol = decision.symbol
    option_type = "CALL" if decision.action == "BUY_CALL" else "PUT"
    # Reuse the contract already found during analysis — looking it up a second time can
    # return a different (often worse/illiquid) contract than the one the brain decided on.
    # Only re-find if it's missing or the brain flipped direction (CALL↔PUT).
    if contract is not None and contract.get("option_type") != option_type:
        contract = None
    if contract is None:
        contract = executor.find_option_contract(symbol, option_type, decision.target_dte)
    if not contract:
        logger.info(f"{symbol}: No suitable {option_type} contract found")
        return False

    # Size against the ASK (worst realistic buy price), not the mid — a wide spread
    # otherwise lets the real fill blow past the position cap (ARKK on 2026-06-04 sized
    # on a $3.53 mid but filled $4.85, breaking the 8% cap).
    ask = float(contract.get("ask") or 0)
    price_for_sizing = max(ask, contract["mid_price"]) if ask > 0 else contract["mid_price"]
    estimated_cost = round(price_for_sizing * 100, 2)
    positions_now = executor.get_positions()
    risk_check = risk.check_new_entry(
        symbol=symbol,
        current_equity=account["equity"],
        available_cash=account["cash"],
        num_open_positions=_open_bet_count(positions_now),
        estimated_premium=estimated_cost,
    )

    if not risk_check.approved:
        logger.info(f"{symbol}: Risk blocked — {risk_check.reason}")
        return False

    contracts = risk_check.contracts
    order = executor.buy_option(contract, contracts=contracts, reason=f"conf={decision.confidence:.2f}")
    if order:
        db.create_trade(
            symbol=symbol,
            side="buy",
            contract_symbol=order["occ_symbol"],
            option_type=order["option_type"],
            strategy_type=decision.action,
            strike=order["strike"],
            expiration=order["expiration"],
            dte_at_entry=order["dte"],
            iv_at_entry=order.get("implied_volatility"),
            contracts=contracts,
            entry_price=order["premium_per_contract"] / 100,
            premium_paid=order["total_cost"],
            peak_value=order["total_cost"],
            entry_time=datetime.now(timezone.utc),
            status="open",
            alpaca_order_id=order.get("order_id"),
            strategy_version=STRATEGY_VERSION,
            catalyst_type=decision.catalyst_type,
            sentiment_shelf_life_days=decision.sentiment_shelf_life_days,
            sentiment_score_at_entry=decision.sentiment_score_at_entry,
        )
        logger.success(
            f"✅ {decision.action} {symbol}: {contracts} contract(s) "
            f"@ ${order['premium_per_contract']/100:.2f}/share "
            f"| total=${order['total_cost']:.0f}"
        )
        return True
    return False


def _execute_credit_spread(decision, account: dict, spread: dict = None) -> bool:
    """Size and SELL a defined-risk credit spread. Returns True if filled."""
    symbol = decision.symbol

    # Iron condors are build-ready but their multi-leg exit management lands in Phase 4.
    # Guard here so one can never be opened until that exit path exists (flag also OFF).
    if decision.action == "IRON_CONDOR":
        logger.info(f"{symbol}: IRON_CONDOR routing not enabled for execution yet — skipping")
        return False

    if spread is None:
        spread = executor.find_credit_spread_contracts(symbol, decision.action, decision.target_dte)
    if not spread:
        logger.info(f"{symbol}: No suitable {decision.action} found at execution")
        return False

    positions_now = executor.get_positions()
    risk_check = risk.check_credit_spread_entry(
        symbol=symbol,
        current_equity=account["equity"],
        available_cash=account["cash"],
        num_open_positions=_open_bet_count(positions_now),
        credit_per_spread=spread["credit_per_spread"],
        width_per_spread=spread["width_per_spread"],
    )
    if not risk_check.approved:
        logger.info(f"{symbol}: Risk blocked (credit spread) — {risk_check.reason}")
        return False

    contracts = risk_check.contracts
    order = executor.sell_credit_spread(spread, contracts=contracts, reason=f"conf={decision.confidence:.2f}")
    if not order:
        return False

    db.create_trade(
        symbol=symbol,
        side="sell",
        contract_symbol=order["short_occ"],     # short leg is the position of record
        long_leg_symbol=order["long_occ"],
        option_type=spread["option_type"],
        strategy_type=decision.action,
        strike=spread["short_leg"]["strike"],
        expiration=order["expiration"],
        dte_at_entry=order["dte"],
        iv_at_entry=decision.options_flow_score,  # placeholder; IV stored via signal snapshot
        contracts=contracts,
        entry_price=order["credit_per_spread"] / 100,
        premium_paid=0.0,                         # credit spread: we COLLECT, not pay
        credit_received=order["total_credit"],
        spread_width=order["width_per_spread"],
        peak_value=0.0,
        entry_time=datetime.now(timezone.utc),
        status="open",
        alpaca_order_id=order.get("order_id"),
        strategy_version=STRATEGY_VERSION,
        catalyst_type=decision.catalyst_type,
        sentiment_shelf_life_days=decision.sentiment_shelf_life_days,
        sentiment_score_at_entry=decision.sentiment_score_at_entry,
    )
    logger.success(
        f"✅ {decision.action} {symbol}: {contracts} spread(s) "
        f"| credit=${order['total_credit']:.0f} | max loss=${risk_check.total_max_loss:.0f}"
    )
    return True


def _to_date(val):
    """Coerce a DB entry_time (datetime or str) to a date; None if unparseable."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _compute_stock_exit_inputs(symbol: str, option_type: str, entry_time, current_price: float):
    """
    Smart stop-loss inputs (Layers 1 & 2), judged off the UNDERLYING stock — not the
    noisy option mark. Returns a dict {thesis_hard, thesis_soft, chandelier_breached,
    technical} or None if data is unavailable (caller then falls back to the old stops).

    Layer 1 (thesis re-validation): re-run the entry screen on the stock.
      CALL thesis intact = stock above its 50-EMA with momentum up. We flag:
        - trend_broken      : stock back below the 50-EMA (the gate it passed at entry)
        - momentum down     : MACD rolled over (daily for the HARD combo; daily-or-hourly soft)
      HARD invalidation = trend_broken AND daily momentum down  → exit now (any P&L).
      SOFT deterioration = trend_broken OR momentum down        → tighten the loss cut.
    Layer 2 (ATR chandelier): stock fell > chandelier_atr_mult × ATR below its highest
      point since entry (mirror for puts). ATR auto-scales the cushion per name.
    """
    if not config.options.enable_smart_stops:
        return None
    try:
        df = pipeline.get_historical_bars(symbol, period_days=90)
        if df is None or len(df) < 55:
            return None
        tech = compute_technical_signals(symbol, df)
        if tech is None or tech.ema_50 is None:
            return None

        # Hourly MACD for faster (soft) momentum confirmation; fails open (bullish).
        h_bull = True
        try:
            hourly = pipeline.get_hourly_bars(symbol)
            if hourly is not None:
                h_bull, _, _ = check_hourly_macd(hourly)
        except Exception:
            pass

        is_call = option_type == "call"
        price = current_price or tech.close
        daily_momentum_down = not tech.macd_bullish
        hourly_momentum_down = not h_bull

        if is_call:
            trend_broken = price < tech.ema_50
            momentum_down_hard = daily_momentum_down
            momentum_down_soft = daily_momentum_down or hourly_momentum_down
        else:  # PUT — mirror: thesis is the stock FALLING
            trend_broken = price > tech.ema_50
            momentum_down_hard = tech.macd_bullish
            momentum_down_soft = tech.macd_bullish or h_bull

        thesis_hard = bool(trend_broken and momentum_down_hard)
        thesis_soft = bool(trend_broken or momentum_down_soft)

        # ── Layer 2: ATR chandelier on the stock (needs a known entry date) ──
        chandelier_breached = False
        atr = tech.atr or 0.0
        entry_date = _to_date(entry_time)
        if atr > 0 and entry_date is not None:
            try:
                sub = df[df.index.date >= entry_date]
                if len(sub) == 0:
                    sub = df.tail(1)
            except Exception:
                sub = df.tail(1)
            mult = config.options.chandelier_atr_mult
            if is_call:
                stock_high = max(float(sub["high"].max()), price)
                chandelier_breached = price < (stock_high - mult * atr)
            else:
                stock_low = min(float(sub["low"].min()), price)
                chandelier_breached = price > (stock_low + mult * atr)

        return {
            "thesis_hard": thesis_hard,
            "thesis_soft": thesis_soft,
            "chandelier_breached": chandelier_breached,
            "technical": tech,
        }
    except Exception as e:
        logger.debug(f"{symbol}: smart-stop inputs failed: {e}")
        return None


def _run_loser_gutcheck(trade, tech, loss_pct: float, days_held: int, dte_remaining: int):
    """Layer 3: pull fresh news + sentiment and ask GPT-4o if the thesis is broken.
    Returns (should_cut, reason). Fails safe (no cut) on any error."""
    try:
        news = pipeline.get_company_news(trade.symbol, hours_back=config.strategy.news_lookback_hours)
    except Exception:
        news = []
    headlines = [n.get("headline", "") for n in news if n.get("headline")]
    sent_score, sent_count = 0.0, 0
    try:
        sd = score_sentiment(trade.symbol, news)
        sent_score, sent_count = sd.overall_score, sd.news_count
    except Exception:
        pass
    return brain.review_losing_position(
        symbol=trade.symbol,
        option_type=(trade.option_type or "call").lower(),
        loss_pct=loss_pct,
        days_held=days_held,
        dte_remaining=dte_remaining,
        technical=tech,
        news_headlines=headlines,
        sentiment_score=sent_score,
        sentiment_news_count=sent_count,
    )


def _check_credit_spread_exit(trade):
    """
    Exit logic for a SOLD credit spread — inverted vs a long option: we collected a credit
    and want the spread to decay toward zero so we can buy it back cheap.
      cost_to_close = (short_leg_price − long_leg_price) × 100 × contracts
    Take profit at 50% of credit captured, stop at 2× credit, time-exit at 21 DTE, and cut
    on a confirmed thesis break (the underlying moved against the spread's direction).
    """
    short_occ = trade.contract_symbol
    long_occ = trade.long_leg_symbol
    contracts = trade.contracts or 1

    short_pos = executor.get_position(short_occ)
    if not short_pos:
        logger.warning(f"Credit spread short leg gone for {short_occ} — marking closed")
        db.update_trade(trade.id, status="closed", exit_reason="position_not_found",
                        exit_time=datetime.now(timezone.utc))
        _stock_exit_streak.pop(short_occ, None)
        _closed_today.add(trade.symbol)  # traded today and now flat → same-day re-entry cooldown
        return

    long_pos = executor.get_position(long_occ) if long_occ else None
    short_price = float(short_pos["current_price"])
    long_price = float(long_pos["current_price"]) if long_pos else 0.0
    cost_to_close = round((short_price - long_price) * 100 * contracts, 2)
    credit_received = trade.credit_received or 0.0

    dte_remaining = 999
    if trade.expiration:
        exp_date = datetime.strptime(trade.expiration, "%Y-%m-%d").date()
        dte_remaining = max(0, (exp_date - date.today()).days)

    # Thesis break: map the spread to the stock direction it bet on, then reuse the
    # underlying-based smart-stop inputs with the 2-cycle confirmation debounce.
    thesis_hard = False
    if config.options.enable_smart_stops and trade.strategy_type in {"SELL_PUT_SPREAD", "SELL_CALL_SPREAD"}:
        mapped = "call" if trade.strategy_type == "SELL_PUT_SPREAD" else "put"
        try:
            stock_price = pipeline.get_current_price(trade.symbol) or 0.0
        except Exception:
            stock_price = 0.0
        inputs = _compute_stock_exit_inputs(trade.symbol, mapped, trade.entry_time, stock_price)
        if inputs:
            if inputs["thesis_hard"]:
                _stock_exit_streak[short_occ] = _stock_exit_streak.get(short_occ, 0) + 1
            else:
                _stock_exit_streak.pop(short_occ, None)
            thesis_hard = (inputs["thesis_hard"]
                           and _stock_exit_streak.get(short_occ, 0) >= config.options.thesis_break_confirm_cycles)

    exit_signal = risk.check_credit_spread_exit(
        symbol=trade.symbol,
        credit_received=credit_received,
        cost_to_close=cost_to_close,
        dte_remaining=dte_remaining,
        thesis_broken_hard=thesis_hard,
    )
    if not exit_signal.should_exit:
        _credit_stop_streak.pop(short_occ, None)
        return

    reason = exit_signal.reason or ""
    is_stop = "credit_stop" in reason

    # ── Stop-loss reliability guards (mark-based STOP only) ───────────────────
    # The cost-to-close stop is the only exit driven by a possibly-bad live mark.
    # Time-exit (DTE) and thesis-break have their own logic; profit-take is benign.
    if is_stop:
        mins = executor.minutes_since_open()
        if mins is not None and mins < config.options.stop_warmup_minutes:
            logger.info(
                f"{short_occ}: credit_stop seen {mins:.1f}min after open "
                f"(< {config.options.stop_warmup_minutes}min warmup) — ignoring unreliable opening mark"
            )
            return
        _credit_stop_streak[short_occ] = _credit_stop_streak.get(short_occ, 0) + 1
        need = config.options.credit_stop_confirm_cycles
        if _credit_stop_streak[short_occ] < need:
            logger.info(
                f"{short_occ}: credit_stop unconfirmed ({_credit_stop_streak[short_occ]}/{need}) "
                f"— waiting for next reading before cutting"
            )
            return
    else:
        _credit_stop_streak.pop(short_occ, None)

    with _exit_lock:
        if short_occ in _closing_contracts:
            return
        if db.get_open_trade_by_contract(short_occ) is None:
            return
        _closing_contracts.add(short_occ)
    try:
        # Buy the spread back as a marketable LIMIT — pay no more than mid × (1 + buffer).
        # A STOP must exit, so an unfilled limit escalates to market inside close_credit_spread;
        # a profit-take simply retries next cycle. (short_price/long_price are the live marks.)
        limit_debit_ps = round((short_price - long_price) * (1 + config.options.spread_exit_slippage_pct), 2)
        order = executor.close_credit_spread(
            short_occ, long_occ, contracts, reason=reason,
            limit_debit_per_share=limit_debit_ps if limit_debit_ps > 0 else None,
            allow_market_fallback=is_stop,
        )
        if not order:
            # Close failed outright — leave the trade OPEN so the next cycle retries.
            # NEVER mark a trade closed that we didn't actually close (the stranded-leg bug).
            logger.warning(f"{trade.symbol}: spread close failed — keeping trade OPEN to retry next cycle")
            return
        if order.get("pdt_blocked"):
            return  # retry next trading day

        # ── Safety net: only mark closed once BOTH legs are confirmed flat ────
        if not order.get("already_closed"):
            residual_short = executor.get_position(short_occ)
            residual_long = executor.get_position(long_occ) if long_occ else None
            if residual_short or residual_long:
                # MLEG should close both atomically; if a leg lingers, close it directly
                # (safe now — it's alone, so there's no uncovered window) and re-check.
                for occ, present in ((short_occ, residual_short), (long_occ, residual_long)):
                    if present and occ:
                        executor.close_option(occ, reason=f"residual leg cleanup | {reason}")
                residual_short = executor.get_position(short_occ)
                residual_long = executor.get_position(long_occ) if long_occ else None
                if residual_short or residual_long:
                    logger.warning(
                        f"{trade.symbol}: spread close left a residual leg "
                        f"(short={'Y' if residual_short else 'N'} long={'Y' if residual_long else 'N'}) "
                        f"— keeping trade OPEN to retry next cycle"
                    )
                    return

        # ── Book P&L off ACTUAL fills; fall back to the mark only if fills unknown ──
        actual_cost = order.get("actual_cost_to_close")
        if actual_cost is None:
            actual_cost = cost_to_close
            logger.warning(f"{trade.symbol}: close fills unconfirmed — booking P&L off the mark (estimate)")
        pnl = round(credit_received - actual_cost, 2)
        pnl_pct = round(pnl / credit_received, 4) if credit_received else 0.0
        # Cooldown on EVERY close (win or loss), not just stops. After profit-taking the
        # original XLF on 2026-06-17 the bot re-entered XLF an hour later and lost $182; a
        # same-day, same-name cooldown stops that round-trip churn. Reset each day at open.
        _closed_today.add(trade.symbol)
        db.update_trade(
            trade.id,
            exit_price=round(actual_cost / 100 / contracts, 4) if contracts else 0.0,
            exit_time=datetime.now(timezone.utc),
            status="closed",
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(
            f"{emoji} EXIT {trade.symbol} ({trade.strategy_type}): {reason} "
            f"| credit ${credit_received:.0f} − close ${actual_cost:.0f} = P&L ${pnl:+.0f} ({pnl_pct:+.0%})"
        )
        _stock_exit_streak.pop(short_occ, None)
        _credit_stop_streak.pop(short_occ, None)
    finally:
        with _exit_lock:
            _closing_contracts.discard(short_occ)


def _check_option_exit(trade):
    """Check all exit conditions for one open options position."""
    if not trade.contract_symbol:
        return

    # Guard against race condition: run_agent_cycle and monitor_positions both fire
    # at the same second every 30 min. Re-query DB to confirm trade is still open
    # before acting — prevents duplicate close orders if the other job beat us to it.
    fresh = db.get_open_trade_by_contract(trade.contract_symbol)
    if fresh is None:
        return

    # Credit spreads (sold premium) use inverted exit math — handle separately.
    if trade.strategy_type in {"SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "IRON_CONDOR"}:
        _check_credit_spread_exit(trade)
        return

    is_spread = trade.strategy_type and "SPREAD" in trade.strategy_type

    # Get long leg position (always required)
    long_position = executor.get_position(trade.contract_symbol)
    if not long_position:
        logger.warning(f"Position not found for {trade.contract_symbol} — marking closed")
        db.update_trade(trade.id, status="closed", exit_reason="position_not_found",
                        exit_time=datetime.now(timezone.utc))
        return

    long_value = float(long_position["current_price"]) * 100 * trade.contracts

    if is_spread:
        # For spreads: net value = long leg value minus short leg liability
        short_occ = _find_spread_short_leg(trade.symbol, trade.contract_symbol)
        short_value = 0.0
        if short_occ:
            short_position = executor.get_position(short_occ)
            if short_position:
                short_value = float(short_position["current_price"]) * 100 * trade.contracts
        current_value = long_value - short_value
    else:
        current_value = long_value

    # Update peak value in DB
    db.update_trade_peak(trade.id, current_value)
    peak_value = max(trade.peak_value or trade.premium_paid, current_value)

    # Calculate DTE remaining
    dte_remaining = 999
    if trade.expiration:
        from datetime import date
        exp_date = datetime.strptime(trade.expiration, "%Y-%m-%d").date()
        dte_remaining = max(0, (exp_date - date.today()).days)

    # Days held: derived from dte_at_entry minus current dte
    days_held = 0
    if trade.dte_at_entry and dte_remaining < 999:
        days_held = max(0, trade.dte_at_entry - dte_remaining)

    # Smart exit params — only fetch for plain long calls/puts (not spreads)
    # Fetching stock price adds one API call per non-spread position per 2-min cycle.
    smart_stock_price = 0.0
    if not is_spread:
        try:
            smart_stock_price = pipeline.get_current_price(trade.symbol) or 0.0
        except Exception:
            pass

    # ── Smart stop-loss inputs (Layers 1 & 2, judged off the underlying stock) ──
    # Confirmation debounce: an immediate stock-based exit (hard thesis break or
    # chandelier) must persist thesis_break_confirm_cycles consecutive monitor cycles
    # before it fires — filters out a single intraday wick that snaps back.
    thesis_hard = thesis_soft = chandelier_breached = False
    tech_state = None
    if not is_spread and config.options.enable_smart_stops:
        inputs = _compute_stock_exit_inputs(
            trade.symbol, (trade.option_type or "call").lower(),
            trade.entry_time, smart_stock_price,
        )
        if inputs:
            tech_state = inputs["technical"]
            thesis_soft = inputs["thesis_soft"]
            raw_immediate = inputs["thesis_hard"] or inputs["chandelier_breached"]
            key = trade.contract_symbol
            if raw_immediate:
                _stock_exit_streak[key] = _stock_exit_streak.get(key, 0) + 1
            else:
                _stock_exit_streak.pop(key, None)
            confirmed = _stock_exit_streak.get(key, 0) >= config.options.thesis_break_confirm_cycles
            thesis_hard = inputs["thesis_hard"] and confirmed
            chandelier_breached = inputs["chandelier_breached"] and confirmed

    # Sentiment decay check — only re-fetch when shelf life has expired for buzz plays.
    # Keeps monitoring cheap: sentiment API only called once per position after shelf life.
    current_sentiment_score = None
    sentiment_news_count = 0
    if (
        trade.catalyst_type == "sentiment_buzz"
        and trade.sentiment_shelf_life_days
        and days_held >= trade.sentiment_shelf_life_days
    ):
        try:
            news = pipeline.get_company_news(trade.symbol, hours_back=config.strategy.news_lookback_hours)
            sent_data = score_sentiment(trade.symbol, news)
            current_sentiment_score = sent_data.overall_score
            sentiment_news_count = sent_data.news_count
            logger.debug(
                f"{trade.symbol}: Sentiment re-check (day {days_held}, shelf={trade.sentiment_shelf_life_days}d): "
                f"score={current_sentiment_score:+.2f}, articles={sentiment_news_count}"
            )
        except Exception as e:
            logger.debug(f"Sentiment re-fetch failed for {trade.symbol}: {e}")

    exit_signal = risk.check_option_exit(
        symbol=trade.symbol,
        premium_paid=trade.premium_paid or 0,
        current_value=current_value,
        dte_remaining=dte_remaining,
        peak_value=peak_value,
        stock_price=smart_stock_price,
        strike=trade.strike or 0.0,
        iv=trade.iv_at_entry or 0.0,
        option_type=(trade.option_type or "call").lower(),
        days_held=days_held,
        contracts=trade.contracts or 1,
        thesis_broken_hard=thesis_hard,
        thesis_deteriorating=thesis_soft,
        chandelier_breached=chandelier_breached,
        catalyst_type=trade.catalyst_type or "",
        sentiment_shelf_life_days=trade.sentiment_shelf_life_days or 0,
        sentiment_score_at_entry=trade.sentiment_score_at_entry or 0.0,
        current_sentiment_score=current_sentiment_score,
        sentiment_news_count=sentiment_news_count,
    )

    # ── Layer 3: AI gut-check on a loser (once per contract) ──────────
    # Only spend a GPT call when the cheaper rules did NOT already fire and the position
    # has genuinely fallen to the trigger loss. Reviews fresh news + the live technical
    # picture and cuts if the thesis is broken (vs. riding it to the -50% floor).
    prem = trade.premium_paid or 0
    if (
        not exit_signal.should_exit
        and not is_spread
        and config.options.enable_smart_stops
        and tech_state is not None
        and prem > 0
        and trade.contract_symbol not in _gutchecked_contracts
    ):
        loss_pct_now = (prem - current_value) / prem
        if loss_pct_now >= config.options.gutcheck_loss_trigger:
            _gutchecked_contracts.add(trade.contract_symbol)
            cut, reason = _run_loser_gutcheck(trade, tech_state, loss_pct_now, days_held, dte_remaining)
            if cut:
                exit_signal = ExitSignal(
                    should_exit=True,
                    reason=f"gutcheck_cut: {reason}",
                    urgency="immediate",
                )

    if exit_signal.should_exit:
        # Claim this contract so the other scheduler thread can't also try to close it
        # (same-second race → duplicate sell → Alpaca "uncovered" rejection).
        with _exit_lock:
            if trade.contract_symbol in _closing_contracts:
                return
            # Re-confirm still open now that we hold the lock (the other thread may have
            # just closed it and written the DB between our earlier check and here).
            if db.get_open_trade_by_contract(trade.contract_symbol) is None:
                return
            _closing_contracts.add(trade.contract_symbol)

        try:
            if is_spread:
                short_occ = _find_spread_short_leg(trade.symbol, trade.contract_symbol)
                if short_occ:
                    order = executor.close_spread(trade.contract_symbol, short_occ, reason=exit_signal.reason)
                else:
                    order = executor.close_option(trade.contract_symbol, reason=exit_signal.reason)
            else:
                order = executor.close_option(trade.contract_symbol, reason=exit_signal.reason)

            if order:
                if order.get("pdt_blocked"):
                    # PDT protection: bought and sold same security same day. Position stays
                    # open in DB and Alpaca. Will be retried on the next trading day.
                    return
                # Same-day cooldown on every close (win or loss), not just stops.
                _closed_today.add(trade.symbol)
                pnl = current_value - (trade.premium_paid or 0)
                pnl_pct = pnl / trade.premium_paid if trade.premium_paid else 0
                db.update_trade(
                    trade.id,
                    exit_price=float(long_position["current_price"]),
                    exit_time=datetime.now(timezone.utc),
                    status="closed",
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 4),
                    exit_reason=exit_signal.reason,
                )
                emoji = "🟢" if pnl > 0 else "🔴"
                logger.info(
                    f"{emoji} EXIT {trade.symbol} ({trade.strategy_type}): "
                    f"{exit_signal.reason} | P&L=${pnl:+.0f} ({pnl_pct:+.0%})"
                )
                # Position is closed — drop its smart-stop bookkeeping.
                _stock_exit_streak.pop(trade.contract_symbol, None)
                _gutchecked_contracts.discard(trade.contract_symbol)
        finally:
            with _exit_lock:
                _closing_contracts.discard(trade.contract_symbol)


def _log_portfolio(equity, cash, positions, daily_pnl: float, circuit_breaker: bool = False):
    invested = sum(p.get("market_value", 0) for p in positions)
    total_pnl = equity - config.risk.initial_capital
    total_pnl_pct = total_pnl / config.risk.initial_capital if config.risk.initial_capital else 0
    # Daily % is vs start-of-day equity (= equity − daily_pnl), matching the loss-limit calc
    start_of_day_equity = equity - daily_pnl
    daily_pnl_pct = daily_pnl / start_of_day_equity if start_of_day_equity else 0

    db.log_portfolio_snapshot(
        equity=equity,
        cash=cash,
        invested_value=invested,
        num_positions=len(positions),
        daily_pnl=round(daily_pnl, 2),
        daily_pnl_pct=round(daily_pnl_pct, 4),
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 4),
        positions_json=positions,
        circuit_breaker_active=circuit_breaker,
    )


def monitor_positions():
    """
    Runs every 2 minutes. Checks ONLY open option positions for exit conditions.
    No LLM calls — pure price math. Catches stops/targets between 15-min entry scans.
    This is what makes options exits fast enough to capture profit spikes.
    """
    if not executor.is_market_open():
        return

    open_trades = db.get_open_trades()
    open_contracts = _known_spread_contracts(open_trades)
    positions = executor.get_positions()
    has_orphans = any(
        "option" in p.get("asset_class", "").lower() and p["symbol"] not in open_contracts
        for p in positions
    )

    if not open_trades and not has_orphans:
        return

    for trade in open_trades:
        try:
            _check_option_exit(trade)
        except Exception as e:
            logger.error(f"Monitor: exit check failed for {trade.contract_symbol}: {e}")

    # Orphan check every 2 min as well — catches DB-missing positions quickly
    positions = executor.get_positions()
    for pos in positions:
        if "option" in pos.get("asset_class", "").lower() and pos["symbol"] not in open_contracts:
            try:
                _close_orphan_if_needed(pos)
            except Exception as e:
                logger.error(f"Monitor orphan check failed for {pos['symbol']}: {e}")


def pre_market_scan():
    """Pre-market sentiment pre-warm: top 10 watchlist items."""
    logger.info("Pre-market sentiment pre-warm...")
    for symbol in config.strategy.watchlist[:10]:
        try:
            news = pipeline.get_company_news(symbol, hours_back=12)
            if news:
                score_sentiment(symbol, news)  # Populates the 2hr cache
        except Exception as e:
            logger.warning(f"{symbol}: Pre-market scan failed: {e}")


def daily_reset():
    """Reset daily risk flags and record opening equity at market open."""
    global _closed_today
    _closed_today = set()
    _candidate_first_seen.clear()  # fresh anti-chase baselines for the new session
    account = executor.get_account()
    risk.reset_daily_flags(current_equity=account["equity"])
    logger.info(f"Daily reset complete. Opening equity: ${account['equity']:,.2f}")


def weekly_feedback_review():
    """Every Friday at 4:15 PM ET — diagnose the week, no parameter changes."""
    logger.info("═══ Weekly Feedback Review ═══")
    try:
        report = tuner.run_weekly_review()
        issues = report.get("top_issues", [])
        flags = report.get("red_flags", [])
        logger.info(f"Weekly review done. Issues: {len(issues)}, Red flags: {len(flags)}")
        if flags:
            for f in flags:
                logger.warning(f"🚩 Red flag: {f}")
    except Exception as e:
        logger.error(f"Weekly feedback review failed: {e}")


def monthly_parameter_review():
    """First Monday of each month at 4:30 PM ET — full analysis + optional parameter tuning."""
    now = datetime.now(pytz.timezone("America/New_York"))
    # Only run on the first Monday of the month (day 1-7, weekday 0=Monday)
    if not (now.weekday() == 0 and 1 <= now.day <= 7):
        return
    logger.info("═══ Monthly Parameter Review ═══")
    try:
        report = tuner.run_monthly_review()
        changed = report.get("parameters_changed", False)
        applied = report.get("applied_changes", [])
        decay = report.get("alpha_decay_status", "UNKNOWN")
        logger.info(f"Monthly review done. Alpha decay: {decay}. Changes applied: {len(applied)}")
        if changed:
            for c in applied:
                logger.info(f"   📐 {c['parameter']}: {c['old_value']} → {c['new_value']}")
    except Exception as e:
        logger.error(f"Monthly parameter review failed: {e}")


def shutdown(signum, frame):
    logger.info("Shutting down...")
    scheduler.shutdown(wait=False)
    sys.exit(0)


if __name__ == "__main__":
    logger.info("🚀 AI Investor Agent v2.0 Starting (Options Mode)...")
    if not config.validate():
        logger.error("Fix config errors above before starting.")
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Pre-market sentiment warm-up at 8:45 AM ET
    scheduler.add_job(pre_market_scan, CronTrigger(day_of_week="mon-fri", hour=8, minute=45, timezone=NY_TZ))
    # Daily flag reset at 9:30 AM ET (market open)
    scheduler.add_job(daily_reset, CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=NY_TZ))
    # Weekly performance diagnosis — every Friday at 4:15 PM ET (after close)
    scheduler.add_job(weekly_feedback_review, CronTrigger(day_of_week="fri", hour=16, minute=15, timezone=NY_TZ))
    # Monthly parameter review — runs every Monday at 4:30 PM ET, guard inside filters to first Monday
    scheduler.add_job(monthly_parameter_review, CronTrigger(day_of_week="mon", hour=16, minute=30, timezone=NY_TZ))
    # Position exit monitor every 2 min — catches stops/targets between entry scans
    scheduler.add_job(
        monitor_positions,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute=f"*/{config.scheduler.position_monitor_minutes}",
            timezone=NY_TZ,
        ),
    )
    # Main entry scan every 15 min from 9 AM – 3:45 PM ET
    scheduler.add_job(
        run_agent_cycle,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute=f"*/{config.scheduler.scan_interval_minutes}",
            timezone=NY_TZ,
        ),
    )

    mode = "📄 PAPER" if config.alpaca.paper else "💰 LIVE"
    logger.info(
        f"Mode: {mode} | Capital: ${config.risk.initial_capital:,.0f} | "
        f"Watchlist: {len(config.strategy.watchlist)} symbols | "
        f"Scan: every {config.scheduler.scan_interval_minutes}min"
    )
    logger.info("Running. Ctrl+C to stop.")
    scheduler.start()
