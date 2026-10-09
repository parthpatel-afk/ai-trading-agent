"""
AI Investor Agent - Configuration v2.0
Options trading: dual-mode (directional + spreads), $10k account.
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AlpacaConfig:
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    base_url: str = (
        "https://paper-api.alpaca.markets"
        if os.getenv("ALPACA_PAPER", "true").lower() == "true"
        else "https://api.alpaca.markets"
    )


@dataclass
class FinnhubConfig:
    api_key: str = os.getenv("FINNHUB_API_KEY", "")


@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    sentiment_model: str = os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-4o-mini")
    brain_model: str = os.getenv("OPENAI_BRAIN_MODEL", "gpt-4o")


@dataclass
class GeminiConfig:
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


@dataclass
class GroqConfig:
    api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@dataclass
class OptionsConfig:
    """Options-specific trading parameters."""
    # Target DTE range for entering new positions
    min_dte: int = 28           # Never buy with < 28 DTE (need buffer before 21 DTE close)
    max_dte: int = 45           # Sweet spot: enough time, not overpriced

    # Target delta for directional plays (ATM-ish)
    min_delta: float = 0.35     # Lower = cheaper but less likely to profit
    max_delta: float = 0.55     # Higher = more expensive but higher probability

    # IV Rank thresholds (0-100 scale)
    iv_rank_cheap: float = 30.0       # Below → buy options (IV underpriced)
    iv_rank_expensive: float = 50.0   # Above → use spreads (IV overpriced for buying)

    # Exit rules
    profit_target_long: float = 1.00  # 100% gain on long options → close (2x premium)
    profit_target_spread: float = 0.70  # 70% of max profit on spreads → close
    stop_loss_pct: float = 0.50       # Exit if option loses 50% of premium paid
    time_exit_dte: int = 21           # Close any position with ≤ 21 DTE remaining (expert 21 DTE rule)

    # Spread configuration
    spread_width_pct: float = 0.03    # Short leg is 3% OTM from long leg

    # Earnings blackout
    earnings_blackout_days: int = 7   # Skip stocks with earnings ≤ 7 days away

    # Liquidity filters
    min_open_interest: int = 100      # Minimum open interest on contract
    min_volume: int = 10              # Minimum daily volume
    max_bid_ask_spread_pct: float = 0.12  # Max bid-ask as % of mid price
    # Hard floor: even when liquidity filters are relaxed (no contract passed the
    # strict filters), never buy a contract below this open interest. An OI=9 ARKK
    # contract slipped through the relaxed path on 2026-06-04 and filled 37% above
    # its quoted mid. Below this floor we skip the trade entirely instead of buying junk.
    min_open_interest_floor: int = 50
    # When buying, cap the price we're willing to pay at ask × (1 + this). Turns the
    # market order into a marketable limit order so a wide/illiquid spread can't fill
    # at a runaway price far above the quote.
    max_fill_slippage_pct: float = 0.05

    # ── Smart stop-loss (layered, underlying-based) — 2026-06-07 ──────────────
    # Replaces reliance on the flat 50% premium stop (which let ARKK ride to -51%
    # before cutting). All four layers judge off the UNDERLYING stock, not the noisy
    # option mark, so they cut real downtrends early without false-firing on bad marks.
    enable_smart_stops: bool = True
    # Layer 1 (thesis re-validation): when a call's thesis DETERIORATES (stock back
    # below its 50-EMA OR momentum flipped down) AND it's underwater, tighten the loss
    # cut from stop_loss_pct (0.50) to this. Hard invalidation (BOTH broken) exits now.
    soft_stop_loss_pct: float = 0.30
    # Layer 2 (ATR chandelier on the stock): exit when the stock falls more than
    # chandelier_atr_mult × ATR below its highest point since entry (mirror for puts).
    # ATR auto-scales per name (TSLA gets room, XLF stays tight). 2.5 ≈ standard swing.
    chandelier_atr_mult: float = 2.5
    # Layer 3 (AI gut-check): the first time a position crosses this loss, run a one-off
    # fresh-news + GPT-4o thesis review; cut immediately if the story is broken.
    gutcheck_loss_trigger: float = 0.25
    # Confirmation: a stock-based immediate exit (hard thesis break / chandelier) must
    # persist this many consecutive monitor cycles before firing (filters intraday wicks).
    thesis_break_confirm_cycles: int = 2

    # ── Anti-chase / extension guard — 2026-06-07 ────────────────────────────
    # Stops the bot buying an at-the-money option right after a name has already run
    # (the 2026-06-03 AAPL chase: gated by stale sentiment, then bought ATM at the high
    # and gave back ~$150). Extension is measured in DAILY-ATR multiples of how far the
    # stock has run since it first appeared as a candidate today — volatility-normalized
    # so jumpy names get more room than calm ones (standard "extended-from-the-mean" method).
    enable_chase_guard: bool = True
    # Mildly extended → don't buy ATM at the high; step to a slightly OTM, further-dated
    # contract (more room to breakeven, slower theta) instead of skipping the trend entirely.
    chase_atr_step: float = 0.5
    chase_step_otm_pct: float = 0.02   # buy ~2% out-of-the-money instead of ATM
    chase_step_dte: int = 50           # and further-dated than the usual 35 DTE
    # Heavily extended → it's a chase; skip and wait for a pullback ("if you missed it,
    # let it go" — chasing raises risk and cuts reward).
    chase_atr_skip: float = 1.25

    # ── CREDIT SPREADS — sell defined-risk premium (MASTER_STRATEGY.md CORE) ──────
    # The strategic pivot: when IV is RICH, stop skipping (or buying inflated premium)
    # and instead SELL a defined-risk credit spread to harvest the volatility risk
    # premium. ENABLED 2026-06-10 for the paper run (run 3). Set False to revert to the
    # old buy-calls behavior. Also turns the entry screen TWO-SIDED (hunts downtrends).
    enable_credit_spreads: bool = True
    # Iron condors (sell BOTH a put spread and a call spread when neutral + rich IV).
    # Separate sub-toggle so one-sided spreads can be enabled/tested first.
    enable_iron_condor: bool = False
    # Target SHORT-leg delta — ~0.30 = tastytrade sweet spot (~70% chance of expiring worthless)
    # Higher delta = collects more premium, making the 1/3 credit-to-width ratio achievable.
    credit_short_delta: float = 0.30
    # Width between short and long strike, as % of the underlying price. The long leg is
    # this far beyond the short leg (further OTM) and caps the loss. Tighter widths give a
    # HIGHER credit/width ratio AND a smaller max loss (better for a $25k account). ~1% of
    # price snaps to roughly a 1–5pt wide spread on liquid names. Tune vs live fills.
    credit_spread_width_pct: float = 0.01
    # Quality gate: require credit ≥ this fraction of the width (never risk too lopsided).
    # 0.20 = accept the standard ~20–27% c/w that 0.30Δ index/large-cap verticals actually
    # print in rich IV (the 0.30 "1/3 rule" target almost never fills on liquid names →
    # blocked every trade on 2026-06-10: AAPL/GOOGL/IWM rich-IV spreads at 21–27% all
    # rejected). With the 50%-profit management rule, ~20%+ c/w is positive-expectancy.
    # Nudge back toward 0.22–0.25 if too many thin spreads appear. Tune vs live fills.
    min_credit_to_width: float = 0.20
    # Exit: take profit when we can buy the spread back for ≤ (1 − target) × credit.
    credit_profit_target: float = 0.50   # close at 50% of max credit captured
    # Exit: stop when the cost to close ≥ this multiple of the credit received.
    credit_stop_multiple: float = 2.0    # cost-to-close ≥ 2× credit → cut (≈ −1× credit)
    # Stop-loss reliability guards (added 2026-06-12 after NVDA stopped on a junk opening
    # mark: $215 cost-to-close that was really ~$105). Mark-based STOPS only:
    stop_warmup_minutes: int = 5         # ignore credit-stops in the first N min after open
    credit_stop_confirm_cycles: int = 2  # stop must read over the line N cycles in a row to fire

    # ── Spread order execution: LIMIT orders, not market (added 2026-06-17) ──────────
    # Root cause of the 2026-06-17 bleed: spreads were opened/closed with MARKET orders,
    # which cross BOTH legs' bid-ask. Real fills came in 25–54% below the quoted mid credit
    # (XLF $169→$78, AMZN $120→$85, NVDA $158→$120, TSLA $109→$80). The halved credit then
    # made the 2× stop trip almost immediately → instant stop-outs (XLF cut 3 min after
    # open for −$182). Fix: price every spread off the mid as a marketable LIMIT order, and
    # SKIP rather than fill at a runaway price. The mid is the fair value; we never give up
    # more than the small buffers below.
    # ENTRY: accept a net credit no worse than mid × (1 − this); below it, cancel and skip.
    spread_entry_slippage_pct: float = 0.05
    # EXIT: pay a net debit no worse than mid × (1 + this) to buy the spread back. On a STOP
    # (must get out) an unfilled limit escalates to market; on a profit-take it just retries.
    spread_exit_slippage_pct: float = 0.10
    # Seconds to wait for a spread limit order to fill before cancelling / escalating.
    spread_fill_wait_seconds: int = 10

    # ── IV environment thresholds (loosened 2026-06-10 after the first paper run) ───
    # SELL premium when the PRIMARY condition holds (IV high vs its own year AND implied
    # richer than realized) — OR a single-signal high-conviction OVERRIDE fires. The strict
    # AND-only version left obvious trades behind on 2026-06-10 (NVDA IVR 92 but IV/HV ~1.0;
    # SPY IV/HV 1.38 but IVR 49 — both skipped). The override captures those: a screamingly
    # high IV-Rank OR a very rich implied-vs-realized ratio is enough on its own.
    iv_rank_sell_min: float = 45.0       # primary: IV-Rank ≥ 45 (was 50) ...
    iv_hv_ratio_sell_min: float = 1.15   # ... AND implied ≥ 1.15× realized (was 1.20)
    iv_rank_sell_override: float = 80.0  # OR IV-Rank ≥ 80 alone → rich (e.g. NVDA 92)
    iv_hv_ratio_sell_override: float = 1.35  # OR IV/HV ≥ 1.35 alone → rich (e.g. SPY 1.38)
    # BUY premium (directional satellite) only when options are genuinely cheap.
    iv_rank_buy_max: float = 25.0
    iv_hv_ratio_buy_max: float = 1.0     # implied ≤ realized → underpriced, buyable
    # Per-trade risk cap for defined-risk spreads: fraction of account equity that the
    # spread's MAX LOSS may consume. Lower than the long-option premium cap because we
    # run more, smaller positions (that's how premium selling wins).
    max_loss_pct_per_trade: float = 0.02  # ~$500 max loss per spread on $25k

    # ── Two-sided + reversal satellite + profit-lock (Phase 4, MASTER_STRATEGY §3.3/§7) ──
    # When the credit engine is on, the entry screen also hunts DOWNtrends (→ bear call
    # spreads / puts), not just uptrends. enable_reversal gates the directional BUY sleeve
    # (cheap-IV calls/puts) to confirmed reversal/exhaustion setups so we stop buying tops.
    # ENABLED 2026-06-10 for the paper run (run 3).
    enable_reversal: bool = True
    # Directional satellite buys are ITM (stock-like, low theta) and further-dated, NOT
    # naked ATM/OTM — the "buy ITM, never the lottery ticket" rule.
    satellite_itm_pct: float = 0.05   # buy ~5% in-the-money (≈ 0.65–0.70 delta)
    satellite_dte: int = 50           # longer-dated than the 35 DTE credit default
    # Profit-lock for long options: take the full position at a big gain instead of waiting
    # for a trailing give-back (fixes SPY +116%→+65%). Tightens nothing else.
    enable_profit_lock: bool = True
    profit_lock_gain: float = 1.00    # close a long option outright at +100%


@dataclass
class StrategyConfig:
    """
    v2.0 Strategy: Options Dual-Mode (Directional Buys + Defined-Risk Spreads)
    """
    # === WATCHLIST: ETFs first (tightest option spreads), then high-liquidity stocks ===
    watchlist: List[str] = field(default_factory=lambda: [
        # ETFs — tightest bid-ask spreads, institutional liquidity
        "SPY", "QQQ", "IWM", "GLD", "XLF", "XLE", "XLK", "XLV", "XLY", "TLT", "GDX", "ARKK",
        # Mega-cap tech — most liquid options in the market
        "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "AMD", "NFLX",
        # Financials
        "JPM", "BAC",
        # Semiconductors / AI infrastructure
        "MU", "MRVL", "SMCI",
        # Cybersecurity
        "PANW", "CRWD",
        # Fintech / Crypto
        "COIN", "PYPL",
        # Consumer / Growth
        "UBER", "DIS",
        # High-conviction speculative
        "SOFI", "PLTR", "HOOD",
    ])

    # === CREDIT-SPREAD CORE UNIVERSE (MASTER_STRATEGY.md §8.1) ===
    # Tightened, deliberately UNCORRELATED, liquid names where credit spreads behave
    # well (penny/nickel-wide options). Selling premium lives or dies on liquidity —
    # a wide option bid-ask eats the credit. The thin/speculative watchlist names are
    # intentionally excluded from the credit CORE (they may still serve directional buys).
    # Only used when enable_credit_spreads is True.
    credit_core_universe: List[str] = field(default_factory=lambda: [
        # Index ETFs — the backbone (tightest spreads, deepest liquidity)
        "SPY", "QQQ", "IWM", "DIA",
        # Uncorrelated diversifiers (low correlation to the index ETFs)
        "GLD", "TLT", "XLE", "XLF",
        # Most-liquid mega-caps (penny/near-penny-wide options)
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    ])

    # === TECHNICAL INDICATOR PARAMETERS ===
    rsi_period: int = 14
    rsi_min: float = 30.0       # Widened: catch more setups without chasing extremes
    rsi_max: float = 78.0
    rsi_exit: float = 78.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    ema_short: int = 9
    ema_long: int = 21
    ema_trend: int = 50

    bollinger_period: int = 20
    bollinger_std: float = 2.0

    atr_period: int = 14
    volume_ma_period: int = 20
    volume_multiplier: float = 0.7       # Relaxed: allows early-day scanning when volume is still building

    # === SENTIMENT ===
    sentiment_entry_min: float = 0.15
    sentiment_exit_max: float = -0.30
    news_lookback_hours: int = 8
    sentiment_cache_minutes: int = 120

    # === ENTRY CONDITIONS ===
    # MACD: no longer requires fresh crossover — just needs MACD > signal (positive momentum)
    # Hourly MACD is checked separately in the entry pipeline for precise timing
    require_above_ema_trend: bool = True
    require_volume_confirmation: bool = False  # Daily bars = partial day volume during trading hours — already in score (10% weight)

    # === SIGNAL WEIGHTS — must sum to 1.0 ===
    weight_technical: float = 0.40
    weight_sentiment: float = 0.25
    weight_macro: float = 0.15
    weight_options_flow: float = 0.20

    # === MINIMUM COMPOSITE SCORE TO TRADE ===
    min_signal_confidence: float = 0.50  # Lowered: let the AI decide, not pre-filter aggressively


@dataclass
class RiskConfig:
    """Hard risk rules — options edition, $10k account."""
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "10000"))
    max_positions: int = int(os.getenv("MAX_POSITIONS", "6"))

    # Per-trade sizing: 8% of account = $800 max per option position
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.08"))

    # Dollar limits per contract
    min_premium: float = float(os.getenv("MIN_PREMIUM", "75"))    # Min $75/contract
    max_premium: float = float(os.getenv("MAX_PREMIUM", "800"))   # Max $800/contract

    # Always keep 20% cash undeployed
    cash_reserve_pct: float = float(os.getenv("CASH_RESERVE_PCT", "0.20"))

    # Halt new entries if we're down 3% on the day
    daily_loss_limit_pct: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.08"))

    # Circuit breaker: stop all trading if portfolio drops 15% from peak
    portfolio_drawdown_limit_pct: float = float(os.getenv("PORTFOLIO_DRAWDOWN_LIMIT_PCT", "0.15"))

    kelly_fraction: float = 0.50  # Half-Kelly for conservative sizing

    # VIX regime thresholds
    vix_high_threshold: float = 25.0
    vix_extreme_threshold: float = 35.0


@dataclass
class SchedulerConfig:
    scan_interval_minutes: int = int(os.getenv("SCAN_INTERVAL_MINUTES", "15"))
    position_monitor_minutes: int = int(os.getenv("POSITION_MONITOR_MINUTES", "2"))
    market_open: str = os.getenv("MARKET_OPEN", "09:30")
    market_close: str = os.getenv("MARKET_CLOSE", "15:45")
    timezone: str = "America/New_York"
    pre_market_scan: bool = True


@dataclass
class Config:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    finnhub: FinnhubConfig = field(default_factory=FinnhubConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./agent_data.db")
    dashboard_api_host: str = os.getenv("DASHBOARD_API_HOST", "localhost")
    dashboard_api_port: int = int(os.getenv("DASHBOARD_API_PORT", "8000"))

    def validate(self) -> bool:
        errors = []
        if not self.alpaca.api_key:
            errors.append("ALPACA_API_KEY is not set")
        if not self.alpaca.secret_key:
            errors.append("ALPACA_SECRET_KEY is not set")
        if not self.openai.api_key:
            errors.append("OPENAI_API_KEY is not set")

        s = self.strategy
        weight_sum = s.weight_technical + s.weight_sentiment + s.weight_macro + s.weight_options_flow
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(f"Signal weights sum to {weight_sum:.3f}, must be 1.0")

        if errors:
            for e in errors:
                print(f"❌ Config Error: {e}")
            return False
        return True


config = Config()
