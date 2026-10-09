"""
AI Investor Agent - Risk Engine v2.0
Options-specific risk management:
- 50% premium loss → exit (options stop loss)
- 7 DTE remaining → time exit (avoid theta acceleration)
- Delta floor < 0.25 → exit (low probability of recovery)
- 21 DTE intrinsic check → exit if intrinsic < 30% of premium paid
- Velocity check → exit if losing faster than 1.5× theta rate for 3+ days
- Tiered trailing stop → lets winners run, tightens as gains grow
- Real daily P&L tracking
- Risk settings read from SQLite (cross-process persistence)
"""
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple
from loguru import logger

from agent.config import config


# ── Options Greeks Helpers ────────────────────────────────────────────────────

def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Black-Scholes delta using only the math module (no scipy dependency).
    S=stock price, K=strike, T=years to expiry, r=risk-free rate, sigma=IV.
    Returns delta in [0,1] for calls, [-1,0] for puts. Returns 0.0 on bad inputs.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))  # standard normal CDF
        return nd1 if option_type.lower() == "call" else nd1 - 1.0
    except Exception:
        return 0.0


def _theta_pct_per_day(dte: int) -> float:
    """
    Estimated % of remaining option premium consumed by theta per day.
    Piecewise linear interpolation from quantified research (daystoexpiry.com):
      45 DTE → 3.3%/day | 30 DTE → 6.7%/day | 21 DTE → 11.1%/day
      14 DTE → 15.0%/day | 7 DTE → 25.0%/day
    """
    if dte >= 45:
        return 0.033
    if dte >= 30:
        return 0.033 + (45 - dte) / 15.0 * (0.067 - 0.033)
    if dte >= 21:
        return 0.067 + (30 - dte) / 9.0 * (0.111 - 0.067)
    if dte >= 14:
        return 0.111 + (21 - dte) / 7.0 * (0.150 - 0.111)
    if dte >= 7:
        return 0.150 + (14 - dte) / 7.0 * (0.250 - 0.150)
    return 0.250


@dataclass
class RiskCheck:
    """Result of a pre-trade risk evaluation."""
    approved: bool
    reason: str
    adjusted_position_pct: float = 0.0
    max_premium_dollars: float = 0.0    # Max dollars to spend on this option
    contracts: int = 1                  # Number of contracts (always int for options)
    stop_loss_pct: float = 0.50
    profit_target_pct: float = 1.00
    # Credit-spread sizing (defined-risk): populated by check_credit_spread_entry
    max_loss_per_spread: float = 0.0    # (width − credit) × 100, dollars per spread
    credit_per_spread: float = 0.0      # credit received per spread, dollars
    total_max_loss: float = 0.0         # contracts × max_loss_per_spread


@dataclass
class ExitSignal:
    """Result of an exit evaluation for an open options position."""
    should_exit: bool
    reason: str
    urgency: str = "normal"  # "normal" | "immediate" (immediate = market order now)


class RiskEngine:
    """
    Enforces all hard risk rules before any option order is placed.
    Reads settings from DB so dashboard changes propagate without restart.
    """

    def __init__(self):
        self._circuit_breaker_active = False
        self._daily_loss_triggered = False
        self._peak_equity: Optional[float] = None
        self._daily_start_equity: Optional[float] = None
        self._daily_reset_date: Optional[date] = None
        # Confirmation counter: a health breach must be seen on 2 consecutive checks
        # before it trips the (sticky) circuit breaker / daily-loss flag. Prevents a
        # single bad mark — e.g. unreliable option prices at the 9:30 open, which falsely
        # showed −14.5% on 2026-06-05 then recovered to −5% one cycle later — from
        # halting trading for the whole day.
        self._unhealthy_streak = 0
        logger.info("Risk Engine v2.0 initialized")

    # ── State management ──────────────────────────────────────────────

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def reset_daily_flags(self, current_equity: float):
        """Call at market open each day. Records opening equity for daily P&L calc."""
        self._daily_loss_triggered = False
        self._daily_start_equity = current_equity
        self._daily_reset_date = date.today()
        logger.info(f"Daily flags reset. Opening equity: ${current_equity:,.2f}")

    def manually_reset_circuit_breaker(self, db=None):
        self._circuit_breaker_active = False
        if db:
            db.save_risk_settings(circuit_breaker_active=False)
        logger.warning("Circuit breaker manually reset by user")

    def sync_settings_from_db(self, db):
        """
        Load risk settings from the DB so that changes made in the dashboard
        take effect in the scheduler process without restarting.
        """
        try:
            settings = db.get_risk_settings()
            if settings:
                config.risk.max_positions = settings.max_positions
                config.risk.max_position_pct = settings.max_position_pct
                config.risk.cash_reserve_pct = settings.cash_reserve_pct
                config.risk.daily_loss_limit_pct = settings.daily_loss_limit_pct
                config.risk.portfolio_drawdown_limit_pct = settings.portfolio_drawdown_limit_pct
                config.risk.min_premium = settings.min_premium
                config.risk.max_premium = settings.max_premium
                logger.debug("Risk settings synced from DB")
            # If dashboard reset the circuit breaker, pick it up here
            if self._circuit_breaker_active:
                latest = db.get_latest_portfolio()
                if latest and not latest.circuit_breaker_active:
                    self._circuit_breaker_active = False
                    logger.warning("Circuit breaker cleared via dashboard reset")
        except Exception as e:
            logger.debug(f"Risk settings DB sync skipped: {e}")

    # ── Portfolio-level circuit breakers ─────────────────────────────

    def check_portfolio_health(
        self,
        current_equity: float,
        daily_pnl: float,          # Mark-to-market $ change today (equity − prior session close)
        num_open_positions: int,
        db=None,
    ) -> Tuple[bool, str]:
        """
        Run on every scan cycle. Checks drawdown and daily loss.
        Returns (is_healthy, message).
        """
        cfg = config.risk

        # Update all-time equity peak
        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # ── Identify any breach this cycle (drawdown OR daily loss) ───────
        drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity

        # Daily loss limit — % of START-OF-DAY equity (how a desk books a daily loss),
        # not a fixed initial-capital figure. start_of_day = current_equity − daily_pnl,
        # since daily_pnl is measured from it.
        start_of_day_equity = current_equity - daily_pnl
        daily_pnl_pct = daily_pnl / start_of_day_equity if start_of_day_equity > 0 else 0.0

        breach_kind = None
        breach_msg = ""
        if drawdown_pct >= cfg.portfolio_drawdown_limit_pct:
            breach_kind = "drawdown"
            breach_msg = (
                f"🚨 CIRCUIT BREAKER: Portfolio down {drawdown_pct:.1%} from peak "
                f"(limit: {cfg.portfolio_drawdown_limit_pct:.1%}). All new entries halted."
            )
        elif daily_pnl_pct <= -cfg.daily_loss_limit_pct:
            breach_kind = "daily_loss"
            breach_msg = (
                f"⚠️ DAILY LOSS LIMIT: Down ${abs(daily_pnl):.0f} today ({daily_pnl_pct:.1%}). "
                f"No new entries until tomorrow. Limit: -{cfg.daily_loss_limit_pct:.1%}"
            )

        # ── No breach → reset confirmation streak, all clear ──────────────
        if breach_kind is None:
            self._unhealthy_streak = 0
            return True, "Portfolio healthy"

        # ── Breach seen — require 2 consecutive confirmations before halting ──
        # A lone bad mark (common at the open on illiquid options) won't trip the
        # sticky flags; a genuine sustained loss will confirm on the next cycle.
        self._unhealthy_streak += 1
        if self._unhealthy_streak < 2:
            logger.warning(
                f"Health breach ({breach_kind}) seen but UNCONFIRMED — awaiting next "
                f"cycle before halting (likely a transient mark). {breach_msg}"
            )
            return True, f"Unconfirmed {breach_kind} breach — not halting yet"

        # Confirmed across 2 checks — trip the appropriate sticky flag
        if breach_kind == "drawdown":
            self._circuit_breaker_active = True
            if db:
                db.save_risk_settings(circuit_breaker_active=True)
            logger.critical(breach_msg)
        else:
            self._daily_loss_triggered = True
            logger.warning(breach_msg)
        return False, breach_msg

    # ── New entry risk checks ─────────────────────────────────────────

    def check_new_entry(
        self,
        symbol: str,
        current_equity: float,
        available_cash: float,
        num_open_positions: int,  # noqa: used for max_positions check
        estimated_premium: float,    # Estimated cost per contract in dollars
    ) -> RiskCheck:
        """
        Validate a proposed new options entry.
        Returns RiskCheck with approved=True/False and adjusted sizing.
        """
        cfg = config.risk

        if self._circuit_breaker_active:
            return RiskCheck(approved=False, reason="Circuit breaker active — entries halted")

        if self._daily_loss_triggered:
            return RiskCheck(approved=False, reason="Daily loss limit triggered — no new entries today")

        if num_open_positions >= cfg.max_positions:
            return RiskCheck(
                approved=False,
                reason=f"Max positions reached ({num_open_positions}/{cfg.max_positions})"
            )

        # ── Cash reserve check ─────────────────────────────────────────
        reserved_cash = current_equity * cfg.cash_reserve_pct
        tradeable_cash = available_cash - reserved_cash
        if tradeable_cash < cfg.min_premium:
            return RiskCheck(
                approved=False,
                reason=f"Insufficient tradeable cash (${tradeable_cash:.0f} after {cfg.cash_reserve_pct:.0%} reserve)"
            )

        # ── Position size cap ──────────────────────────────────────────
        max_position_dollars = current_equity * cfg.max_position_pct
        max_premium = min(max_position_dollars, cfg.max_premium, tradeable_cash * 0.95)

        if max_premium < cfg.min_premium:
            return RiskCheck(
                approved=False,
                reason=f"Max allowed premium ${max_premium:.0f} below minimum ${cfg.min_premium:.0f}"
            )

        # ── Contract count ─────────────────────────────────────────────
        # If 1 contract already costs more than our max, reject the trade.
        # Never force through an oversized position just to trade something.
        if estimated_premium > max_premium:
            return RiskCheck(
                approved=False,
                reason=(
                    f"1 contract costs ${estimated_premium:.0f}, exceeds max position "
                    f"of ${max_premium:.0f} ({cfg.max_position_pct:.0%} of equity) — skipping"
                )
            )

        contracts = int(max_premium // max(estimated_premium, 1))
        contracts = max(1, contracts)
        actual_cost = contracts * estimated_premium

        if actual_cost > tradeable_cash:
            return RiskCheck(
                approved=False,
                reason=f"Insufficient cash: need ${actual_cost:.0f}, have ${tradeable_cash:.0f}"
            )

        position_pct = actual_cost / current_equity

        logger.info(
            f"Risk approved: {symbol} | {contracts} contract(s) | "
            f"cost=${actual_cost:.0f} ({position_pct:.1%} of equity)"
        )

        return RiskCheck(
            approved=True,
            reason="All risk checks passed",
            adjusted_position_pct=round(position_pct, 4),
            max_premium_dollars=round(actual_cost, 2),
            contracts=contracts,
            stop_loss_pct=cfg.max_position_pct,  # placeholder, brain sets actual
            profit_target_pct=1.00,
        )

    # ── Credit-spread entry (defined-risk, sized on MAX LOSS not premium) ──

    def check_credit_spread_entry(
        self,
        symbol: str,
        current_equity: float,
        available_cash: float,
        num_open_positions: int,
        credit_per_spread: float,   # dollars received per spread (credit_per_share × 100)
        width_per_spread: float,    # dollars between strikes per spread (width_pts × 100)
    ) -> RiskCheck:
        """
        Validate and size a defined-risk credit spread.

        Unlike a long option (sized on premium PAID), a credit spread is sized on its
        MAX LOSS = (width − credit) × 100. We risk at most `max_loss_pct_per_trade` of
        equity per spread position, and require credit ≥ `min_credit_to_width` × width so
        we never risk $3 to make $1.
        """
        cfg = config.risk
        cfg_opts = config.options

        if self._circuit_breaker_active:
            return RiskCheck(approved=False, reason="Circuit breaker active — entries halted")
        if self._daily_loss_triggered:
            return RiskCheck(approved=False, reason="Daily loss limit triggered — no new entries today")
        if num_open_positions >= cfg.max_positions:
            return RiskCheck(
                approved=False,
                reason=f"Max positions reached ({num_open_positions}/{cfg.max_positions})",
            )

        if credit_per_spread <= 0 or width_per_spread <= 0:
            return RiskCheck(approved=False, reason="Invalid credit/width data")

        max_loss_per_spread = round(width_per_spread - credit_per_spread, 2)
        if max_loss_per_spread <= 0:
            return RiskCheck(approved=False, reason="Spread has no defined max loss (credit ≥ width?)")

        # Quality gate: credit must be a worthwhile fraction of the width.
        if credit_per_spread < cfg_opts.min_credit_to_width * width_per_spread:
            return RiskCheck(
                approved=False,
                reason=(
                    f"Credit ${credit_per_spread:.0f} < {cfg_opts.min_credit_to_width:.0%} of "
                    f"width ${width_per_spread:.0f} — risk/reward too poor"
                ),
            )

        # Risk budget: at most max_loss_pct_per_trade of equity may be at risk on this spread.
        risk_budget = current_equity * cfg_opts.max_loss_pct_per_trade
        if max_loss_per_spread > risk_budget:
            return RiskCheck(
                approved=False,
                reason=(
                    f"One spread risks ${max_loss_per_spread:.0f} > per-trade budget "
                    f"${risk_budget:.0f} ({cfg_opts.max_loss_pct_per_trade:.0%} of equity)"
                ),
            )

        contracts = int(risk_budget // max_loss_per_spread)

        # Defined-risk margin (cash held) = max loss × contracts. Respect the cash reserve.
        reserved_cash = current_equity * cfg.cash_reserve_pct
        tradeable_cash = available_cash - reserved_cash
        while contracts >= 1 and contracts * max_loss_per_spread > tradeable_cash:
            contracts -= 1
        if contracts < 1:
            return RiskCheck(
                approved=False,
                reason=(
                    f"Insufficient tradeable cash for 1 spread: need ${max_loss_per_spread:.0f}, "
                    f"have ${tradeable_cash:.0f} after {cfg.cash_reserve_pct:.0%} reserve"
                ),
            )

        total_max_loss = round(contracts * max_loss_per_spread, 2)
        total_credit = round(contracts * credit_per_spread, 2)
        logger.info(
            f"Risk approved (credit spread): {symbol} | {contracts} spread(s) | "
            f"credit=${total_credit:.0f} | max loss=${total_max_loss:.0f} "
            f"({total_max_loss / current_equity:.1%} of equity)"
        )
        return RiskCheck(
            approved=True,
            reason="Credit-spread risk checks passed",
            contracts=contracts,
            adjusted_position_pct=round(total_max_loss / current_equity, 4),
            max_loss_per_spread=max_loss_per_spread,
            credit_per_spread=round(credit_per_spread, 2),
            total_max_loss=total_max_loss,
        )

    def check_credit_spread_exit(
        self,
        symbol: str,
        credit_received: float,   # total dollars collected at entry (all contracts)
        cost_to_close: float,     # current dollars to buy the spread back (all contracts)
        dte_remaining: int,
        thesis_broken_hard: bool = False,  # underlying broke the direction the spread bet on
    ) -> ExitSignal:
        """
        Exit logic for a SOLD credit spread — the mirror image of a long option.
        We collected `credit_received` and want the spread to decay toward zero so we can
        buy it back cheap. `cost_to_close` is what it costs to buy it back right now.

          profit = credit_received − cost_to_close
          • take profit when cost_to_close ≤ (1 − target) × credit   (default: 50% captured)
          • stop loss   when cost_to_close ≥ stop_multiple × credit   (default: 2× = −1× credit)
          • time exit at 21 DTE (correct for sellers — dodge end-of-life gamma/pin risk)
          • thesis break: the underlying moved against the spread's direction → cut now
        """
        cfg_opts = config.options
        if credit_received <= 0:
            return ExitSignal(should_exit=False, reason="Invalid credit data")

        # ── Time exit: 21 DTE ─────────────────────────────────────────
        if dte_remaining <= cfg_opts.time_exit_dte:
            return ExitSignal(
                should_exit=True,
                reason=f"time_exit: {dte_remaining} DTE ≤ {cfg_opts.time_exit_dte} (gamma/pin risk)",
                urgency="immediate",
            )

        # ── Thesis break: the move went against us → cut before max loss ──
        if thesis_broken_hard:
            return ExitSignal(
                should_exit=True,
                reason="thesis_broken: underlying moved against the spread — cutting the short side",
                urgency="immediate",
            )

        # ── Profit target: bought back cheap enough ───────────────────
        profit_take_level = (1.0 - cfg_opts.credit_profit_target) * credit_received
        if cost_to_close <= profit_take_level:
            captured = (credit_received - cost_to_close) / credit_received
            return ExitSignal(
                should_exit=True,
                reason=(
                    f"credit_profit: captured {captured:.0%} of credit "
                    f"(buy back ${cost_to_close:.0f} vs ${credit_received:.0f} collected)"
                ),
                urgency="normal",
            )

        # ── Stop loss: cost to close ballooned ────────────────────────
        if cost_to_close >= cfg_opts.credit_stop_multiple * credit_received:
            loss = cost_to_close - credit_received
            return ExitSignal(
                should_exit=True,
                reason=(
                    f"credit_stop: cost to close ${cost_to_close:.0f} ≥ "
                    f"{cfg_opts.credit_stop_multiple:g}× credit ${credit_received:.0f} "
                    f"(loss ${loss:.0f})"
                ),
                urgency="immediate",
            )

        return ExitSignal(should_exit=False, reason="Credit spread within parameters")

    # ── Options-specific exit checks ──────────────────────────────────

    def check_option_exit(
        self,
        symbol: str,
        premium_paid: float,
        current_value: float,
        dte_remaining: int,
        peak_value: float,
        # Smart exit params — optional, checks only run when provided
        stock_price: float = 0.0,    # Current underlying stock price
        strike: float = 0.0,        # Option strike price
        iv: float = 0.0,            # Implied volatility (decimal, e.g. 0.45)
        option_type: str = "call",  # "call" or "put"
        days_held: int = 0,         # Calendar days position has been open
        contracts: int = 1,         # kept for API compatibility, reserved for future use
        # Smart stop-loss params (computed in scheduler off the underlying stock) ──
        thesis_broken_hard: bool = False,   # Layer 1 HARD: trend broken AND momentum flipped (confirmed)
        thesis_deteriorating: bool = False, # Layer 1 SOFT: one warning sign (trend OR momentum)
        chandelier_breached: bool = False,  # Layer 2: stock fell > mult×ATR below its post-entry high (confirmed)
        gutcheck_cut: bool = False,         # Layer 3: AI thesis review said cut
        gutcheck_reason: str = "",
        # Sentiment decay params — only active for sentiment_buzz catalyst trades
        catalyst_type: str = "",
        sentiment_shelf_life_days: int = 0,
        sentiment_score_at_entry: float = 0.0,
        current_sentiment_score: Optional[float] = None,
        sentiment_news_count: int = 0,
    ) -> ExitSignal:
        """
        Check all exit conditions for an open options position.
        New smart checks (delta floor, 21 DTE intrinsic, velocity) activate
        automatically when stock_price/strike/iv/days_held are supplied.
        """
        cfg_opts = config.options
        RISK_FREE_RATE = 0.045  # ~4.5% US Treasury rate

        if premium_paid <= 0:
            return ExitSignal(should_exit=False, reason="Invalid premium data")

        gain_pct = (current_value - premium_paid) / premium_paid
        loss_pct = (premium_paid - current_value) / premium_paid

        # ── Time Exit: 7 DTE (theta acceleration) ─────────────────────
        if dte_remaining <= cfg_opts.time_exit_dte:
            return ExitSignal(
                should_exit=True,
                reason=f"time_exit: {dte_remaining} DTE ≤ {cfg_opts.time_exit_dte} (theta risk)",
                urgency="immediate"
            )

        # ── Smart stop-loss layers (underlying-based) ─────────────────
        # These fire BEFORE the flat 50% floor so a real downtrend is cut early,
        # not ridden all the way to -50% (ARKK bled to -51% on 2026-06-05 because
        # nothing but the flat stop could catch a fast <1-day breakdown).
        # All booleans are computed in the scheduler off the underlying stock and
        # already confirmation-debounced; here we just turn them into exit signals.
        if cfg_opts.enable_smart_stops:

            # Layer 1 — HARD thesis invalidation: the trend that justified the trade
            # has reversed (stock back below 50-EMA AND momentum flipped). The reason
            # to own it is gone → exit at any P&L (locks gains on a winner too).
            if thesis_broken_hard:
                return ExitSignal(
                    should_exit=True,
                    reason=(
                        f"thesis_broken: trend + momentum both reversed on the underlying "
                        f"(now {gain_pct:+.0%}) — reason to hold is gone"
                    ),
                    urgency="immediate",
                )

            # Layer 2 — ATR chandelier: the stock has rolled over from its post-entry
            # high by more than chandelier_atr_mult × normal daily range.
            if chandelier_breached:
                return ExitSignal(
                    should_exit=True,
                    reason=(
                        f"chandelier_stop: underlying fell >{cfg_opts.chandelier_atr_mult:g}×ATR "
                        f"from its high since entry (now {gain_pct:+.0%}) — up-move reversed"
                    ),
                    urgency="immediate",
                )

            # Layer 3 — AI gut-check verdict (only set when a loser crossed the trigger
            # and the fresh-news/GPT-4o review judged the thesis broken).
            if gutcheck_cut:
                return ExitSignal(
                    should_exit=True,
                    reason=f"gutcheck_cut: AI thesis review → cut ({gutcheck_reason or 'thesis broken'})",
                    urgency="immediate",
                )

            # Layer 1 — SOFT deterioration: one warning sign (trend OR momentum) AND
            # underwater → tighten the loss cut from 50% to soft_stop_loss_pct (30%),
            # so a deteriorating trade bleeds half as much before it's cut.
            if thesis_deteriorating and loss_pct >= cfg_opts.soft_stop_loss_pct:
                return ExitSignal(
                    should_exit=True,
                    reason=(
                        f"soft_stop: lost {loss_pct:.0%} with thesis deteriorating "
                        f"(tightened limit: {cfg_opts.soft_stop_loss_pct:.0%})"
                    ),
                    urgency="immediate",
                )

        # ── Stop Loss: 50% of premium paid (hard backstop) ────────────
        if loss_pct >= cfg_opts.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=f"stop_loss: lost {loss_pct:.0%} of premium (limit: {cfg_opts.stop_loss_pct:.0%})",
                urgency="immediate"
            )

        # ── Smart Exit Checks (only for plain long calls/puts, not spreads) ──
        # These only run when the caller passes stock_price + strike.

        # Check 1 — Delta Floor (probability of profit < 25%)
        # A delta below 0.25 means the market prices in <25% chance of recovering.
        # Computing this requires stock_price, strike, iv, and DTE.
        if stock_price > 0 and strike > 0 and iv > 0 and dte_remaining > cfg_opts.time_exit_dte:
            T = dte_remaining / 365.0
            delta = _bs_delta(stock_price, strike, T, RISK_FREE_RATE, iv, option_type)
            abs_delta = abs(delta)
            if abs_delta < 0.25 and abs_delta > 0:
                return ExitSignal(
                    should_exit=True,
                    reason=(
                        f"delta_floor: delta={delta:.2f} < 0.25 "
                        f"(<25% probability of reaching breakeven)"
                    ),
                    urgency="normal"
                )

        # Check 3 — Velocity Deterioration
        # If over the days held the position has been losing more than 1.5× what
        # theta alone explains, the underlying is actively moving against us.
        # Only applies after 3+ days so we have a meaningful trend.
        if days_held >= 3 and current_value > 0 and current_value < premium_paid:
            avg_daily_loss = (premium_paid - current_value) / days_held
            est_theta_dollars = current_value * _theta_pct_per_day(dte_remaining)
            if avg_daily_loss > est_theta_dollars * 1.5:
                return ExitSignal(
                    should_exit=True,
                    reason=(
                        f"velocity_exit: losing ${avg_daily_loss:.0f}/day avg over {days_held}d "
                        f"vs est. theta ${est_theta_dollars:.0f}/day "
                        f"(underlying moving against position, not just time decay)"
                    ),
                    urgency="normal"
                )

        # ── Profit lock: take the whole position at a big gain ────────
        # Fixes the "let a +116% winner sag to +65% before selling" problem (SPY 2026-06-08):
        # instead of always waiting for a trailing give-back, bank a large gain outright.
        if cfg_opts.enable_profit_lock and gain_pct >= cfg_opts.profit_lock_gain:
            return ExitSignal(
                should_exit=True,
                reason=f"profit_lock: +{gain_pct:.0%} ≥ target +{cfg_opts.profit_lock_gain:.0%} — banking the winner",
                urgency="normal",
            )

        # ── Tiered Trailing Stop ───────────────────────────────────────
        # No hard profit cap — let winners run. Trail tightens as gains grow.
        #
        #   Peak gain reached  |  Max pullback before exit
        #   ≥ 150%             |  25%  (very tight — protect a big winner)
        #   ≥ 100%             |  30%  (tight — locked in a double)
        #   ≥  50%             |  40%  (moderate — good profit, give room)
        #   ≥  20%             |  60%  (loose — early profit, don't shake out)
        #   <  20%             |  not active (still in hard-stop-only zone)
        #
        # Pullback % = how much of the total profit from entry to peak has been given back.
        # Example: paid $800, peaked at $2,400 (+200%), now at $1,760 (+120%)
        #   decay = ($2,400 - $1,760) / ($2,400 - $800) = $640 / $1,600 = 40%
        #   Peak tier ≥ 150% → threshold 25% → 40% > 25% → EXIT at +120%

        peak_gain_pct = (peak_value - premium_paid) / premium_paid if peak_value > premium_paid else 0.0

        if peak_gain_pct >= 0.20:
            if peak_gain_pct >= 1.50:
                pullback_threshold = 0.25
            elif peak_gain_pct >= 1.00:
                pullback_threshold = 0.30
            elif peak_gain_pct >= 0.50:
                pullback_threshold = 0.40
            else:
                pullback_threshold = 0.60

            decay_from_peak = (peak_value - current_value) / (peak_value - premium_paid)
            if decay_from_peak >= pullback_threshold:
                return ExitSignal(
                    should_exit=True,
                    reason=(
                        f"trailing_stop: gave back {decay_from_peak:.0%} from peak "
                        f"(peak +{peak_gain_pct:.0%}, now {gain_pct:+.0%}, threshold {pullback_threshold:.0%})"
                    ),
                    urgency="normal"
                )

        # ── Sentiment Decay Exit (buzz plays only) ────────────────────
        # Fires only when ALL guards pass — prevents false sells on quiet news days:
        # 1. Trade was entered on short-term news buzz
        # 2. Shelf life has expired (days_held >= shelf life set by brain)
        # 3. Score dropped meaningfully (>= 0.30 from entry)
        # 4. Current score is below near-neutral floor (< 0.20 on -1..+1 scale)
        # 5. We have enough articles to trust the score (>= 2; prevents 0-article ghost reads)
        # 6. Position is profitable — never force a loss just because news faded
        if (
            catalyst_type == "sentiment_buzz"
            and sentiment_shelf_life_days > 0
            and days_held >= sentiment_shelf_life_days
            and current_sentiment_score is not None
            and sentiment_news_count >= 2
            and (sentiment_score_at_entry - current_sentiment_score) >= 0.30
            and current_sentiment_score < 0.20
            and current_value > premium_paid
        ):
            return ExitSignal(
                should_exit=True,
                reason=(
                    f"sentiment_decay: buzz expired after {days_held}d "
                    f"(entry={sentiment_score_at_entry:+.2f}→current={current_sentiment_score:+.2f}, "
                    f"{sentiment_news_count} articles — locking in profit)"
                ),
                urgency="normal",
            )

        return ExitSignal(should_exit=False, reason="Position within parameters")

    def kelly_position_size(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> float:
        """Half-Kelly fraction of capital. Falls back to 6% with no history."""
        if win_rate <= 0 or avg_win_pct <= 0 or avg_loss_pct <= 0:
            return 0.06

        b = avg_win_pct / avg_loss_pct
        p = win_rate
        q = 1 - p
        full_kelly = (b * p - q) / b
        half_kelly = full_kelly * config.risk.kelly_fraction
        return max(0.03, min(half_kelly, config.risk.max_position_pct))
