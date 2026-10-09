"""
AI Investor Agent - Options Flow Signal Module
Computes IV Rank, checks earnings blackout, selects strategy mode.
This is the 4th signal pillar (20% weight) unique to options trading.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf
import finnhub
from loguru import logger

from agent.config import config


@dataclass
class OptionsFlowSignals:
    symbol: str

    # Implied Volatility
    iv_rank: Optional[float] = None          # 0–100: where current IV sits in its 1-yr range
    iv_current: Optional[float] = None       # Current ATM implied volatility (annualized)
    hv_current: Optional[float] = None       # Current 21d realized (historical) volatility, annualized
    iv_hv_ratio: Optional[float] = None      # iv_current / hv_current (>1 = implied richer than realized)
    iv_regime: str = "unknown"               # "cheap" | "normal" | "expensive"
    iv_environment: str = "unknown"          # "rich" | "cheap" | "middle" — drives buy-vs-sell (credit engine)

    # Earnings risk
    earnings_within_blackout: bool = False
    earnings_date: Optional[str] = None

    # Options market sentiment
    put_call_ratio: Optional[float] = None   # >1.2 = bearish skew, <0.7 = bullish skew

    # Composite score (0–1) and recommended approach
    options_score: float = 0.5
    # "BUY_CALL"|"BUY_PUT"|"SELL_PUT_SPREAD"|"SELL_CALL_SPREAD"|"IRON_CONDOR"|"SKIP"
    strategy_mode: str = "SKIP"
    notes: str = ""


def compute_options_flow(
    symbol: str,
    technical_score: float,
    sentiment_score: float,  # raw -1..+1
    direction: Optional[str] = None,  # "up" | "down" | None(=infer, legacy)
) -> OptionsFlowSignals:
    """
    Compute all options-specific signals for one symbol.
    Uses yfinance (free) for options chains and earnings dates.
    `direction` (from the two-sided screen) drives buy/sell routing when provided.
    """
    signals = OptionsFlowSignals(symbol=symbol)
    cfg_opts = config.options

    try:
        ticker = yf.Ticker(symbol)

        # ── 1. Earnings Calendar ──────────────────────────────────────
        _check_earnings(ticker, signals, cfg_opts.earnings_blackout_days)

        # ── 2. IV Rank + Put/Call Ratio from options chain ────────────
        _compute_iv_and_pcr(ticker, signals)

        # ── 3. IV Rank via Historical Volatility comparison ───────────
        if signals.iv_rank is None:
            _compute_iv_rank_from_hv(ticker, signals)

    except Exception as e:
        logger.warning(f"{symbol}: Options flow computation error: {e}")

    # ── 4. IV Regime classification ───────────────────────────────────
    if signals.iv_rank is not None:
        if signals.iv_rank < cfg_opts.iv_rank_cheap:
            signals.iv_regime = "cheap"
        elif signals.iv_rank > cfg_opts.iv_rank_expensive:
            signals.iv_regime = "expensive"
        else:
            signals.iv_regime = "normal"

    # ── 4b. IV/HV ratio + IV environment (drives buy-vs-sell, credit engine) ──
    # IV/HV > 1 means implied vol is priced richer than the stock is actually moving —
    # the volatility risk premium. We require BOTH a high IV-Rank AND a rich IV/HV ratio
    # before selling premium, so we never "sell cheap vol" (MASTER_STRATEGY.md §4).
    if signals.iv_current and signals.hv_current and signals.hv_current > 0:
        signals.iv_hv_ratio = round(signals.iv_current / signals.hv_current, 2)
    signals.iv_environment = _classify_iv_environment(signals, cfg_opts)

    # ── 5. Options composite score (0–1) ──────────────────────────────
    score = _compute_options_score(signals)
    signals.options_score = score

    # ── 6. Strategy mode selection ────────────────────────────────────
    signals.strategy_mode = _select_strategy(signals, technical_score, sentiment_score, direction)

    logger.info(
        f"{symbol}: IV_Rank={signals.iv_rank}, IV/HV={signals.iv_hv_ratio}, "
        f"IV_Env={signals.iv_environment}, P/C={signals.put_call_ratio}, "
        f"Earnings={signals.earnings_within_blackout}, Mode={signals.strategy_mode}"
    )
    return signals


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_earnings(ticker, signals: OptionsFlowSignals, blackout_days: int):
    """Check earnings proximity. Finnhub is primary (API-based), yfinance is fallback."""

    # ── Finnhub primary (reliable, structured API — not a scraper) ────
    if config.finnhub.api_key:
        try:
            fh = finnhub.Client(api_key=config.finnhub.api_key)
            today_str = datetime.now().strftime("%Y-%m-%d")
            end_str = (datetime.now() + timedelta(days=blackout_days + 2)).strftime("%Y-%m-%d")
            result = fh.earnings_calendar(
                _from=today_str, to=end_str,
                symbol=signals.symbol, international=False,
            )
            for item in result.get("earningsCalendar", []):
                ed_str = item.get("date", "")
                if not ed_str:
                    continue
                days_away = (datetime.strptime(ed_str, "%Y-%m-%d").date() - date.today()).days
                if 0 <= days_away <= blackout_days:
                    signals.earnings_within_blackout = True
                    signals.earnings_date = ed_str
                    signals.notes += f"⚠️ Earnings in {days_away}d ({ed_str}) [Finnhub]. "
                    return
            return  # Finnhub responded cleanly — no fallback needed
        except Exception as e:
            logger.debug(f"{signals.symbol}: Finnhub earnings failed ({e}) — falling back to yfinance")

    # ── yfinance fallback ─────────────────────────────────────────────
    try:
        cal = ticker.calendar
        if cal is None:
            return

        dates_to_check = []
        if isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.columns:
                dates_to_check = cal['Earnings Date'].dropna().tolist()
        elif isinstance(cal, dict):
            raw = cal.get('Earnings Date') or cal.get('earningsDate')
            if raw:
                dates_to_check = raw if isinstance(raw, list) else [raw]

        for ed in dates_to_check:
            try:
                if hasattr(ed, 'to_pydatetime'):
                    ed_dt = ed.to_pydatetime().replace(tzinfo=None)
                else:
                    ed_dt = pd.Timestamp(ed).to_pydatetime().replace(tzinfo=None)
                days_away = (ed_dt.date() - date.today()).days
                if 0 <= days_away <= blackout_days:
                    signals.earnings_within_blackout = True
                    signals.earnings_date = str(ed_dt.date())
                    signals.notes += f"⚠️ Earnings in {days_away}d ({signals.earnings_date}) [yfinance]. "
                    break
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"Earnings calendar unavailable for {signals.symbol}: {e}")


def _compute_iv_and_pcr(ticker, signals: OptionsFlowSignals):
    """Get current ATM IV and put/call ratio from nearest expiration."""
    try:
        opt_dates = ticker.options
        if not opt_dates:
            return

        # Pick expiration closest to our target DTE window
        cfg_opts = config.options
        target_exp = datetime.now().date() + timedelta(days=(cfg_opts.min_dte + cfg_opts.max_dte) // 2)
        best_exp = min(
            opt_dates,
            key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - target_exp).days)
        )

        chain = ticker.option_chain(best_exp)
        calls_df = chain.calls
        puts_df = chain.puts

        # Get current underlying price
        hist = ticker.history(period="1d")
        if hist.empty:
            return
        current_price = float(hist["Close"].iloc[-1])

        if not calls_df.empty and "impliedVolatility" in calls_df.columns:
            calls_df = calls_df.copy()
            calls_df["strike_dist"] = abs(calls_df["strike"] - current_price)
            # Filter for liquid contracts
            liquid = calls_df[
                (calls_df["openInterest"] > cfg_opts.min_open_interest) &
                (calls_df["volume"] > cfg_opts.min_volume)
            ]
            if liquid.empty:
                liquid = calls_df  # fall back to all if liquidity filters too strict
            atm = liquid.loc[liquid["strike_dist"].idxmin()]
            iv_val = atm["impliedVolatility"]
            if not np.isnan(iv_val) and iv_val > 0:
                signals.iv_current = round(float(iv_val), 4)

        # Put/Call ratio from total volume
        call_vol = float(calls_df["volume"].sum()) if not calls_df.empty else 0
        put_vol = float(puts_df["volume"].sum()) if not puts_df.empty else 0
        if call_vol > 0:
            signals.put_call_ratio = round(put_vol / call_vol, 2)

    except Exception as e:
        logger.debug(f"Options chain unavailable for {signals.symbol}: {e}")


def _compute_iv_rank_from_hv(ticker, signals: OptionsFlowSignals):
    """
    Compute IV Rank using 252-day rolling Historical Volatility (HV).

    IV Rank = (current_HV - 52w_HV_low) / (52w_HV_high - 52w_HV_low) x 100

    We intentionally do NOT use signals.iv_current (live ATM option mid-price) here.
    Live option prices fluctuate every second with every bid/ask update, which causes
    IV Rank to swing 7% -> 58% in 90 minutes on the same stock — that is not what
    IV Rank means. True IV Rank is a daily metric; professional platforms compute it
    once at market close from daily closing IV snapshots.

    Since yfinance does not provide historical daily IV snapshots (a paid data product),
    we use rolling 21-day HV from daily CLOSING prices as a proxy. HV is highly
    correlated with IV and updates exactly once per day — giving us a stable rank
    that doesn't create phantom "expensive" rejections from intraday quote noise.

    We also drop today's partial bar (the most recent bar during market hours contains
    only intraday data up to now, not a full day's close) to prevent intraday price
    moves from contaminating the current HV reading.
    """
    try:
        hist = ticker.history(period="252d", interval="1d")
        if hist is None or len(hist) < 60:
            signals.iv_rank = 50.0
            return

        # Drop today's partial intraday bar — HV must be computed from full daily closes only
        if len(hist) > 1:
            hist = hist.iloc[:-1]

        close = hist["Close"].dropna()
        log_returns = np.log(close / close.shift(1)).dropna()
        rolling_hv = log_returns.rolling(21).std() * np.sqrt(252)
        rolling_hv = rolling_hv.dropna()

        if len(rolling_hv) < 30:
            signals.iv_rank = 50.0
            return

        # Always use the most recent full-day HV — never the live chain IV
        current_hv = float(rolling_hv.iloc[-1])
        # Store realized vol so we can compute the IV/HV ratio (VRP signal) later.
        if current_hv > 0:
            signals.hv_current = round(current_hv, 4)
        hv_min = float(rolling_hv.min())
        hv_max = float(rolling_hv.max())

        if hv_max > hv_min:
            signals.iv_rank = round(((current_hv - hv_min) / (hv_max - hv_min)) * 100, 1)
            signals.iv_rank = max(0.0, min(100.0, signals.iv_rank))
        else:
            signals.iv_rank = 50.0

    except Exception as e:
        logger.debug(f"IV Rank HV fallback failed for {signals.symbol}: {e}")
        signals.iv_rank = 50.0


def _compute_options_score(signals: OptionsFlowSignals) -> float:
    """Composite options score 0–1. Higher = better conditions for trading options."""
    score = 0.5  # neutral baseline

    # IV rank: low IV → better to BUY options → higher score for directional plays
    if signals.iv_rank is not None:
        # Score peaks when IV rank is low (cheap options to buy)
        iv_score = 1.0 - (signals.iv_rank / 100.0)
        score = score * 0.4 + iv_score * 0.6

    # Put/Call ratio: extreme readings = signal strength (either direction)
    if signals.put_call_ratio is not None:
        # PCR near 1.0 = neutral; deviation = stronger signal
        pcr_deviation = abs(signals.put_call_ratio - 1.0)
        pcr_score = min(pcr_deviation, 1.0)
        score = score * 0.8 + pcr_score * 0.2

    # Earnings blackout: heavy penalty — binary event = gambling, not trading
    if signals.earnings_within_blackout:
        score *= 0.15

    return round(max(0.0, min(1.0, score)), 3)


def _classify_iv_environment(signals: OptionsFlowSignals, cfg_opts) -> str:
    """
    Classify the IV environment as "rich" (sell premium), "cheap" (buy premium), or
    "middle" (no volatility edge). Requires BOTH conditions for "rich" so we never sell
    cheap vol; "cheap" allows the IV/HV ratio to be unknown (buying is the safer default).
    """
    ivr = signals.iv_rank
    ratio = signals.iv_hv_ratio
    if ivr is None:
        return "unknown"

    # Rich if the PRIMARY both-conditions hold, OR a single-signal high-conviction override:
    #   • IV-Rank screamingly high on its own (e.g. NVDA 92), or
    #   • implied vol very rich vs realized on its own (e.g. SPY IV/HV 1.38).
    primary = ivr >= cfg_opts.iv_rank_sell_min and ratio is not None and ratio >= cfg_opts.iv_hv_ratio_sell_min
    override = (ivr >= cfg_opts.iv_rank_sell_override) or (ratio is not None and ratio >= cfg_opts.iv_hv_ratio_sell_override)
    if primary or override:
        return "rich"

    cheap = ivr <= cfg_opts.iv_rank_buy_max and (ratio is None or ratio <= cfg_opts.iv_hv_ratio_buy_max)
    if cheap:
        return "cheap"

    return "middle"


def _select_strategy(
    signals: OptionsFlowSignals,
    technical_score: float,
    sentiment_score: float,  # -1..+1
    direction: Optional[str] = None,  # "up" | "down" | None(=infer from scores, legacy)
) -> str:
    """
    Decide WHAT type of options play to make based on signal alignment + IV environment.
    Returns: "BUY_CALL" | "BUY_PUT" | "SELL_PUT_SPREAD" | "SELL_CALL_SPREAD"
             | "IRON_CONDOR" | "SKIP"

    Two regimes:
      • Legacy (enable_credit_spreads=False) — unchanged: buy calls/puts in cheap/normal
        IV, SKIP when IV is expensive.
      • Credit engine (enable_credit_spreads=True) — the MASTER_STRATEGY decision matrix:
        rich IV → SELL a defined-risk credit spread (collect the VRP); cheap IV → BUY
        directionally; middle IV → SKIP (sit in cash, no volatility edge).

    When the two-sided screen passes an explicit `direction`, it is authoritative; otherwise
    the legacy score-inference is used (keeps old behavior identical when direction is None).
    """
    cfg_opts = config.options

    # Hard block: never trade through earnings (binary IV-crush risk, both buy and sell)
    if signals.earnings_within_blackout:
        return "SKIP"

    if direction == "up":
        bullish, bearish = True, False
    elif direction == "down":
        bullish, bearish = False, True
    else:
        # Legacy inference — lowered from 0.58 to catch solid setups that aren't perfect.
        # Sentiment allows slightly negative (-0.20) so mixed news doesn't kill a strong setup.
        bullish = technical_score >= 0.50 and sentiment_score >= -0.20
        bearish = technical_score <= 0.35 and sentiment_score <= 0.10

    # ── Credit engine: route by IV environment (sell rich, buy cheap, else cash) ──
    if cfg_opts.enable_credit_spreads:
        env = signals.iv_environment
        if env == "rich":
            if bullish:
                signals.notes += (f"Bullish + IV rich (IVR={signals.iv_rank:.0f}, "
                                  f"IV/HV={signals.iv_hv_ratio}) → SELL put spread (collect premium). ")
                return "SELL_PUT_SPREAD"
            if bearish:
                signals.notes += (f"Bearish + IV rich (IVR={signals.iv_rank:.0f}, "
                                  f"IV/HV={signals.iv_hv_ratio}) → SELL call spread. ")
                return "SELL_CALL_SPREAD"
            if cfg_opts.enable_iron_condor:
                signals.notes += (f"Neutral + IV rich (IVR={signals.iv_rank:.0f}) → iron condor. ")
                return "IRON_CONDOR"
            signals.notes += "IV rich but no directional lean and condors off → SKIP. "
            return "SKIP"
        if env == "cheap":
            if bullish:
                signals.notes += f"Bullish + IV cheap (IVR={signals.iv_rank:.0f}) → buy calls. "
                return "BUY_CALL"
            if bearish:
                signals.notes += f"Bearish + IV cheap (IVR={signals.iv_rank:.0f}) → buy puts. "
                return "BUY_PUT"
            signals.notes += "IV cheap but no directional lean → SKIP. "
            return "SKIP"
        # middle / unknown — no volatility edge in either direction → sit in cash
        signals.notes += f"IV environment '{env}' (IVR={signals.iv_rank}) — no vol edge → SKIP. "
        return "SKIP"

    # ── Legacy directional-buy behavior (credit spreads disabled) ──────
    # Hard block: never BUY options when IV is expensive (IV_Rank > iv_rank_expensive).
    if signals.iv_regime == "expensive":
        signals.notes += f"IV_Rank={signals.iv_rank:.1f}% (expensive) → SKIP directional buy. "
        return "SKIP"

    if bullish:
        signals.notes += f"Bullish + IV {signals.iv_regime} → buy calls. "
        return "BUY_CALL"
    elif bearish:
        signals.notes += f"Bearish + IV {signals.iv_regime} → buy puts. "
        return "BUY_PUT"
    else:
        signals.notes += "No clear directional conviction. "
        return "SKIP"
