"""
Trend-Buyer v1 — Configuration

A mechanical, daily long-options trend-follower:
  buy DEEP-ITM calls in confirmed uptrends / puts in confirmed downtrends,
  only when options are cheap, then ride the trend with a Chandelier trailing
  stop after banking a partial profit. No spreads, no AI (Phase 2).

All trading parameters live here. See docs/NEXT_BOT_DESIGN.md (the spec) and
docs/TRAILING_STOP_CHOICE.md (the exit). Strategy decisions locked 2026-06-27.
"""
import os
import copy
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv

from agent.paths import db_path

load_dotenv()  # project-root .env


@dataclass
class AlpacaConfig:
    """Broker + market-data credentials (the only API the bot needs)."""
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    base_url: str = (
        "https://paper-api.alpaca.markets"
        if os.getenv("ALPACA_PAPER", "true").lower() == "true"
        else "https://api.alpaca.markets"
    )


@dataclass
class StrategyConfig:
    """The trend signal: 9/21 EMA crossover on the daily chart (mirrors the v2 client bot)."""
    # Same 18-name liquid universe as the v2 client bot, so we get a clean A/B read on
    # whether our added filters/IV/exits actually improve on the pure-mechanical baseline.
    watchlist: List[str] = field(default_factory=lambda: [
        "META", "PLTR", "HOOD", "MSFT", "SPY", "IWM", "QQQ", "NVDA", "TSLA",
        "AAPL", "GOOG", "AMZN", "AMD", "TSM", "UNH", "IONQ", "WMT", "JPM",
    ])

    ema_fast: int = 9
    ema_slow: int = 21
    timeframe: str = "1Day"

    # A crossover is "fresh"/actionable for this many completed daily bars.
    signal_lookback_days: int = 2

    # Daily bars to pull. NOTE: this is CALENDAR days, but the filters need ~220 TRADING bars
    # (200-SMA + 20-day slope). 320 calendar days = only ~218 trading bars → 2 short → every
    # symbol was wrongly skipped "insufficient history for filters" (2026-06-29 bug). 500 calendar
    # days ≈ ~345 trading bars: a comfortable margin over the 220 needed.
    history_days: int = 500


@dataclass
class FiltersConfig:
    """Trend-confirmation filters that turn a raw EMA cross into a high-quality signal.
    A raw 9/21 cross wins ~32-50%; these filters lift it toward ~60-68% (research)."""
    # --- ADX: trend-strength gate (0-100; >25 = real trend, <20 = chop/whipsaw) ---
    require_adx: bool = True
    adx_period: int = 14
    adx_min: float = 25.0

    # --- ATR: used for the hard stop distance + the anti-chase extension measure ---
    atr_period: int = 14

    # --- Regime filter: only take CALLS in an up-regime, PUTS in a down-regime ---
    require_regime: bool = True
    regime_trend_sma: int = 200          # price must be on the right side of this
    regime_fast_sma: int = 50            # 50-SMA vs 200-SMA ("golden/death cross" backdrop)
    regime_slope_lookback: int = 20      # 200-SMA must be rising/falling over this many bars

    # --- Anti-chase: don't buy after the move already happened ---
    anti_chase: bool = True
    chase_ref_sma: int = 50              # measure extension from this average
    chase_atr_max: float = 1.25          # skip if price is > this many ATRs past the reference

    # --- Volume confirmation (MED-confidence; toggle) ---
    require_volume: bool = True
    volume_lookback: int = 20            # latest bar volume must exceed this many-day average


@dataclass
class IVConfig:
    """Cheap-vol gate. We proxy 'implied vol cheapness' with the underlying's historical
    volatility (HV) rank — computed from daily closes (reliable, no intraday noise).
    Only buy when vol is in the cheap part of its own 1-year range, so we don't overpay
    the volatility risk premium (the buyer's structural disadvantage)."""
    # Decision #1: we DON'T hard-block expensive vol — we handle it by buying DEEPER ITM
    # (see OptionsConfig.deep_itm_*). The gate below only blocks the genuinely EXTREME top of
    # the range, where even deep-ITM carries too much extrinsic / IV-crush risk. Cheap vol is
    # still preferred (it's the buyer's edge) but no longer required. (HV-rank is a proxy for
    # implied-vol cheapness — tunable.)
    require_cheap_vol: bool = True
    hv_window: int = 20                  # trading days for the HV estimate
    vol_rank_lookback: int = 252         # ~1yr window to rank today's HV within
    vol_rank_max: float = 85.0           # block only extreme vol (0=cheapest, 100=priciest)
    # Sanity bounds on any IV/HV ratio before it can influence anything (ledger lesson).
    iv_hv_min: float = 0.2
    iv_hv_max: float = 3.0


@dataclass
class OptionsConfig:
    """How the bot picks the contract once a signal passes all filters.
    Core rule: DEEP in-the-money (mostly real value, little theta 'ice cube' to melt)."""
    # --- Strike by delta (deep ITM) ---
    target_delta: float = 0.75           # default depth
    delta_min: float = 0.65              # acceptable band floor
    delta_max: float = 0.90              # acceptable band ceiling
    # Decision #1: when vol is expensive, go EVEN DEEPER so there's almost no extrinsic to overpay.
    deep_itm_vol_threshold: float = 40.0   # vol-rank above this => use the deeper target
    deep_itm_target_delta: float = 0.82
    risk_free_rate: float = 0.04         # for the Black-Scholes delta estimate

    # --- Expiry: standard monthly (3rd Friday) closest to ~75 DTE, band 45-120 ---
    prefer_monthly: bool = True
    target_dte: int = 75
    min_dte: int = 45
    max_dte: int = 120

    # --- Liquidity guards (bad fills kill buyers) ---
    min_open_interest: int = 100
    min_volume: int = 10
    max_bid_ask_spread_pct: float = 0.12
    min_open_interest_floor: int = 50

    # --- Entry pricing: limit walked from MID toward a small ceiling, never the raw ask ---
    entry_mid_ceiling_pct: float = 0.08
    entry_walk_steps: int = 3
    entry_walk_wait_seconds: int = 4


@dataclass
class ExitConfig:
    """The layered exit: hard stop (cut losers) -> partial take -> Chandelier trail (ride winners).
    ALL underlying-based, judged on the daily close, with a 2-read confirmation (ledger rule)."""
    # --- Layer 1: initial hard stop (active entry -> partial) ---
    hard_stop_atr_period: int = 14
    hard_stop_atr_mult: float = 2.0          # exit if underlying closes this many ATRs against entry
    use_ema_stop: bool = True                # also exit if it closes back across the 21 EMA
    premium_stop_pct: float = 0.50           # backstop only: option premium down >= 50%
    stop_confirm_reads: int = 2              # consecutive daily closes required before cutting

    # --- Layer 2: partial profit-take (arms the trail) ---
    partial_profit_pct: float = 0.85         # take partial when the option is up >= +85%
    partial_scale_fraction: float = 0.50     # sell this fraction of contracts (needs >=2 to scale)

    # --- Layer 3: Chandelier trailing stop on the runner (see TRAILING_STOP_CHOICE.md) ---
    chandelier_atr_period: int = 22
    chandelier_atr_mult: float = 3.0         # arm A (pick). Arm B for A/B test = 2.5.
    trail_confirm_reads: int = 2             # consecutive closes below the line before exiting

    # --- Layer 4: theta time-cap (never hold into the decay red zone) ---
    time_exit_dte: int = 21


@dataclass
class AccountConfig:
    """Capital, sizing, and portfolio guardrails (carried from the evidence ledger)."""
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "25000"))

    # AGGRESSIVE by design (owner: "risk to make"). Risk per trade = the dollars we'd lose if
    # the HARD STOP hits (NOT the full premium — a deep-ITM option is mostly real value and
    # won't go to zero at the stop). Estimated as delta × stop_distance(ATRs) × 100 × contracts.
    # 4% of equity per trade. The trail + partial + per-name stop cap the downside; the
    # direction/sector caps below are what keep this aggression survivable (no correlated wipeout).
    risk_per_trade_pct: float = 0.04
    # Absolute ceiling: allow a single (chunky deep-ITM) contract through only if its estimated
    # stop-loss is within this % of equity; otherwise REJECT (never force an oversized bet).
    max_risk_per_trade_pct: float = 0.06
    # CAPITAL cap per position: premium deployed in one name must not exceed this % of equity.
    # Deep-ITM ties up real capital (mostly intrinsic), so this is the concentration guard —
    # set high to allow conviction-sized positions. REJECT if even 1 contract breaches it.
    max_position_pct: float = 0.25

    max_positions: int = 6                    # fewer, bigger bets
    # No more than this fraction of total open risk pointed the same way (calls vs puts).
    # The key guardrail that makes the aggressive sizing survivable.
    direction_cap_pct: float = 0.70
    max_per_sector: int = 3
    cash_reserve_pct: float = 0.05           # keep this much cash un-deployed (soft)
    min_contracts: int = 1

    # Rough sector map for the correlation/direction caps (soft guard).
    sectors: Dict[str, str] = field(default_factory=lambda: {
        "NVDA": "semis", "AMD": "semis", "TSM": "semis",
        "MSFT": "megacap_tech", "AAPL": "megacap_tech", "META": "megacap_tech",
        "GOOG": "megacap_tech", "AMZN": "megacap_tech",
        "PLTR": "growth_tech", "IONQ": "growth_tech", "TSLA": "growth_tech", "HOOD": "fintech",
        "JPM": "financials", "UNH": "healthcare", "WMT": "staples",
        "SPY": "index", "QQQ": "index", "IWM": "index",
    })

    def sector_of(self, symbol: str) -> str:
        return self.sectors.get(symbol, "other")


@dataclass
class SchedulerConfig:
    # Entry evaluation shortly after the open (uses the prior session's closed daily bar).
    eval_hour: int = 9
    eval_minute: int = 35
    # Exit/trail/partial monitor near the close (acts on ~final daily prices, before 16:00).
    monitor_hour: int = 15
    monitor_minute: int = 50

    snapshot_interval_minutes: int = 15
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "America/New_York"


@dataclass
class Config:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    iv: IVConfig = field(default_factory=IVConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{db_path().as_posix()}")
    dashboard_api_host: str = os.getenv("DASHBOARD_API_HOST", "localhost")
    dashboard_api_port: int = int(os.getenv("DASHBOARD_API_PORT", "8001"))

    def validate(self) -> bool:
        errors = []
        if not self.alpaca.api_key:
            errors.append("ALPACA_API_KEY is not set")
        if not self.alpaca.secret_key:
            errors.append("ALPACA_SECRET_KEY is not set")
        if self.options.max_dte < self.options.min_dte:
            errors.append("OPTIONS max_dte must be >= min_dte")
        if errors:
            for e in errors:
                print(f"❌ Config Error: {e}")
            return False
        return True

    def clone_with(self, overrides: Dict[str, object]) -> "Config":
        """Return a DEEP COPY of this config with dotted fields overridden — the mechanism
        the backtester's parameter sweeps use to vary one setting while holding the rest fixed.
        Keys are dotted paths into the nested dataclasses, e.g.:
            cfg.clone_with({"filters.adx_min": 20, "iv.vol_rank_max": 60})
        The live singleton is never mutated. Unknown paths raise (fail fast on typos)."""
        clone = copy.deepcopy(self)
        for path, value in overrides.items():
            section, _, attr = path.partition(".")
            if not attr:
                raise KeyError(f"override '{path}' must be 'section.field' (e.g. 'filters.adx_min')")
            obj = getattr(clone, section, None)
            if obj is None or not hasattr(obj, attr):
                raise KeyError(f"unknown config path '{path}'")
            setattr(obj, attr, value)
        return clone


config = Config()
