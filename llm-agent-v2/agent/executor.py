"""
AI Investor Agent - Options Executor v2.0
Handles options contract selection and order placement.

Contract selection: yfinance options chain (free, has delta/IV/OI/volume)
Order execution: Alpaca Trading API (supports options via OCC symbol)

OCC symbol format: AAPL240119C00150000
  = underlying + YYMMDD + C/P + 8-digit strike (strike × 1000, zero-padded)
"""
import time
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict
import numpy as np
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, GetOrdersRequest, OptionLegRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from loguru import logger

from agent.config import config
from agent.risk import _bs_delta


class Executor:
    """
    Wraps Alpaca Trading API for options order placement.
    Uses yfinance to select the best option contract before calling Alpaca.
    """

    def __init__(self):
        self.client = TradingClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            paper=config.alpaca.paper,
        )
        mode = "PAPER" if config.alpaca.paper else "LIVE 💰"
        logger.info(f"Executor v2.0 initialized — Mode: {mode}")

    # ── Account & Positions ──────────────────────────────────────────

    def get_account(self) -> Dict:
        acct = self.client.get_account()
        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),  # equity at prior session close — daily P&L baseline
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "status": str(acct.status),
            "trading_blocked": acct.trading_blocked,
            "paper": config.alpaca.paper,
        }

    def get_positions(self) -> List[Dict]:
        """Return all open positions (stocks + options)."""
        positions = self.client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": str(p.side),
                "asset_class": str(p.asset_class) if hasattr(p, "asset_class") else "us_equity",
            })
        return result

    def get_option_positions(self) -> List[Dict]:
        """Return only options positions."""
        return [p for p in self.get_positions() if p.get("asset_class") == "us_option"]

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            p = self.client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
        except Exception:
            return None

    def is_market_open(self) -> bool:
        try:
            clock = self.client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.warning(f"Market clock check failed: {e}")
            return False

    def minutes_since_open(self) -> Optional[float]:
        """
        Minutes elapsed since today's regular-session open, or None if market is closed
        or the clock can't be read. Used to suppress mark-based stops in the first few
        minutes after the open, when option quotes are widest/least reliable. Derives
        today's open from next_close − 6.5h (regular session), so it needs no TZ library.
        """
        try:
            clock = self.client.get_clock()
            if not clock.is_open:
                return None
            open_t = clock.next_close - timedelta(hours=6, minutes=30)
            return (clock.timestamp - open_t).total_seconds() / 60.0
        except Exception as e:
            logger.warning(f"minutes_since_open check failed: {e}")
            return None

    # ── Option Contract Selection ────────────────────────────────────

    def _find_contract_via_alpaca(
        self,
        symbol: str,
        option_type: str,
        target_dte: int,
        otm_offset_pct: float = 0.0,
        itm_offset_pct: float = 0.0,
    ) -> Optional[Dict]:
        """
        Find the best option contract using Alpaca's data API.
        Primary source — more reliable than yfinance options chains.
        Returns None silently if Alpaca options data is unavailable (SDK version,
        data subscription tier, or network issue) so caller falls back to yfinance.
        """
        try:
            from alpaca.data.historical import OptionHistoricalDataClient
            from alpaca.data.requests import OptionChainRequest
        except ImportError:
            return None

        try:
            cfg_opts = config.options
            data_client = OptionHistoricalDataClient(
                api_key=config.alpaca.api_key,
                secret_key=config.alpaca.secret_key,
            )

            # Get current underlying price via yfinance (very reliable for price, unlike chains)
            hist = yf.Ticker(symbol).history(period="2d")
            if hist.empty:
                return None
            current_price = float(hist["Close"].iloc[-1])

            min_exp = date.today() + timedelta(days=cfg_opts.min_dte)
            max_exp = date.today() + timedelta(days=cfg_opts.max_dte)
            strike_min = round(current_price * 0.85, 2)
            strike_max = round(current_price * 1.15, 2)

            request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                type=option_type.lower(),
                strike_price_gte=str(strike_min),
                strike_price_lte=str(strike_max),
            )
            snapshots = data_client.get_option_chain(request)
            if not snapshots:
                return None

            # Group liquid contracts by expiration date, then pick the expiration
            # closest to target_dte, then find ATM within that expiration.
            target_exp_date = date.today() + timedelta(days=target_dte)
            by_expiration: Dict[date, list] = {}

            for occ_symbol, snap in snapshots.items():
                q = snap.latest_quote
                if not q or not q.bid_price or not q.ask_price:
                    continue
                if float(q.bid_price) <= 0 or float(q.ask_price) <= 0:
                    continue
                mid = (float(q.bid_price) + float(q.ask_price)) / 2
                if mid <= 0:
                    continue
                spread_pct = (float(q.ask_price) - float(q.bid_price)) / mid
                if spread_pct > cfg_opts.max_bid_ask_spread_pct:
                    continue

                # Parse OCC symbol — last 15 chars are always YYMMDD + C/P + 8-digit-strike
                try:
                    strike = int(occ_symbol[-8:]) / 1000
                    exp_date_obj = datetime.strptime(occ_symbol[-15:-9], "%y%m%d").date()
                    dte = (exp_date_obj - date.today()).days
                except Exception:
                    continue

                if not (cfg_opts.min_dte <= dte <= cfg_opts.max_dte):
                    continue

                by_expiration.setdefault(exp_date_obj, []).append({
                    "occ_symbol": occ_symbol,
                    "strike": strike,
                    "dte": dte,
                    "mid": mid,
                    "bid": float(q.bid_price),
                    "ask": float(q.ask_price),
                    "iv": float(snap.implied_volatility) if snap.implied_volatility else 0,
                    "exp_date_obj": exp_date_obj,
                })

            if not by_expiration:
                return None

            # Pick the expiration closest to target_dte
            best_exp = min(by_expiration.keys(), key=lambda d: abs((d - target_exp_date).days))
            candidates = by_expiration[best_exp]

            # Within that expiration, pick the strike closest to the target. Target is
            # ATM (current price) by default; a positive otm_offset_pct steps it out-of-
            # the-money (above price for calls, below for puts) — used by the anti-chase
            # guard to avoid buying ATM right at a local high.
            # net offset: +otm steps further OTM, +itm steps in-the-money (opposite sign)
            sign = 1.0 if option_type.lower() == "call" else -1.0
            target_strike = current_price * (1 + sign * (otm_offset_pct - itm_offset_pct))
            best_row = min(candidates, key=lambda c: abs(c["strike"] - target_strike))
            best = {
                "occ_symbol": best_row["occ_symbol"],
                "underlying": symbol,
                "option_type": option_type,
                "strike": best_row["strike"],
                "expiration": best_row["exp_date_obj"].strftime("%Y-%m-%d"),
                "dte": best_row["dte"],
                "mid_price": round(best_row["mid"], 2),
                "bid": best_row["bid"],
                "ask": best_row["ask"],
                "implied_volatility": best_row["iv"],
                "open_interest": 0,
                "volume": 0,
                "premium_per_contract": round(best_row["mid"] * 100, 2),
            }

            if best:
                logger.info(
                    f"{symbol}: Found {option_type} via Alpaca: {best['occ_symbol']} "
                    f"| strike=${best['strike']:.0f} | DTE={best['dte']} "
                    f"| mid=${best['mid_price']:.2f}"
                )
            return best

        except Exception as e:
            logger.debug(f"{symbol}: Alpaca options chain failed ({e})")
            return None

    def find_option_contract(
        self,
        symbol: str,
        option_type: str,            # "CALL" or "PUT"
        target_dte: int = 35,
        otm_offset_pct: float = 0.0,  # >0 → step strike OTM (anti-chase); 0 = ATM (default)
        itm_offset_pct: float = 0.0,  # >0 → step strike IN-the-money (satellite ITM buys)
    ) -> Optional[Dict]:
        """
        Find the best option contract. Tries Alpaca data API first (reliable),
        falls back to yfinance if Alpaca options data is unavailable.
        Returns contract info dict or None if no suitable contract found.
        """
        # ── Alpaca primary ────────────────────────────────────────────
        result = self._find_contract_via_alpaca(symbol, option_type, target_dte, otm_offset_pct, itm_offset_pct)
        if result:
            return result
        logger.debug(f"{symbol}: Alpaca options unavailable — falling back to yfinance")

        # ── yfinance fallback ─────────────────────────────────────────
        cfg_opts = config.options
        try:
            ticker = yf.Ticker(symbol)
            opt_dates = ticker.options
            if not opt_dates:
                logger.warning(f"{symbol}: No options dates available")
                return None

            # Pick expiration closest to target DTE within our allowed range
            target_exp_date = date.today() + timedelta(days=target_dte)
            valid_dates = [
                d for d in opt_dates
                if cfg_opts.min_dte <= (datetime.strptime(d, "%Y-%m-%d").date() - date.today()).days <= cfg_opts.max_dte
            ]
            if not valid_dates:
                logger.warning(f"{symbol}: No expirations in {cfg_opts.min_dte}-{cfg_opts.max_dte} DTE range")
                return None

            best_exp = min(
                valid_dates,
                key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - target_exp_date).days)
            )

            chain = ticker.option_chain(best_exp)
            df = chain.calls if option_type == "CALL" else chain.puts

            if df is None or df.empty:
                logger.warning(f"{symbol}: Empty {option_type} chain for {best_exp}")
                return None

            # Get current underlying price
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            current_price = float(hist["Close"].iloc[-1])

            df = df.copy()
            # Target strike: ATM by default; otm_offset_pct steps OTM (anti-chase),
            # itm_offset_pct steps in-the-money (satellite ITM buys) — opposite directions.
            sign = 1.0 if option_type == "CALL" else -1.0
            target_strike = current_price * (1 + sign * (otm_offset_pct - itm_offset_pct))
            df["strike_dist"] = abs(df["strike"] - target_strike)

            # Apply liquidity filters
            liquid = df[
                (df["openInterest"] >= cfg_opts.min_open_interest) &
                (df["volume"] >= cfg_opts.min_volume)
            ].copy()

            # Filter bid-ask spread quality
            if "bid" in liquid.columns and "ask" in liquid.columns:
                liquid = liquid[(liquid["bid"] > 0) & (liquid["ask"] > 0)].copy()
                liquid["mid"] = (liquid["bid"] + liquid["ask"]) / 2
                liquid["spread_pct"] = (liquid["ask"] - liquid["bid"]) / liquid["mid"]
                liquid = liquid[liquid["spread_pct"] <= cfg_opts.max_bid_ask_spread_pct]

            if liquid.empty:
                # Relax filters and just find ATM — but keep a HARD open-interest floor
                # so we never buy a near-untraded contract (an OI=9 ARKK call slipped
                # through here on 2026-06-04 and filled 37% above its quoted mid).
                logger.debug(f"{symbol}: Relaxing liquidity filters")
                liquid = df.copy()
                if "openInterest" in liquid.columns:
                    liquid = liquid[liquid["openInterest"] >= cfg_opts.min_open_interest_floor].copy()
                if "bid" in liquid.columns and "ask" in liquid.columns:
                    liquid = liquid[(liquid["bid"] > 0) & (liquid["ask"] > 0)].copy()
                    liquid["mid"] = (liquid["bid"] + liquid["ask"]) / 2
                if liquid.empty:
                    logger.info(
                        f"{symbol}: No contract meets OI floor "
                        f"({cfg_opts.min_open_interest_floor}) — skipping"
                    )
                    return None

            # Select contract closest to ATM (best delta for directional play)
            best = liquid.loc[liquid["strike_dist"].idxmin()]

            strike = float(best["strike"])
            mid_price = float(best.get("mid", best.get("lastPrice", 0)))
            if mid_price <= 0:
                return None

            exp_date = datetime.strptime(best_exp, "%Y-%m-%d").date()
            dte = (exp_date - date.today()).days

            # Build OCC symbol from yfinance contractSymbol (already OCC format)
            occ_symbol = best.get("contractSymbol", "")
            if not occ_symbol:
                occ_symbol = _build_occ_symbol(symbol, exp_date, option_type, strike)

            iv = float(best.get("impliedVolatility", 0)) if not np.isnan(best.get("impliedVolatility", float("nan"))) else 0
            oi_raw = best.get("openInterest", 0)
            oi = int(oi_raw) if oi_raw and not np.isnan(float(oi_raw)) else 0
            vol_raw = best.get("volume", 0)
            vol = int(vol_raw) if vol_raw and not np.isnan(float(vol_raw)) else 0

            logger.info(
                f"{symbol}: Found {option_type} contract {occ_symbol} "
                f"| strike=${strike:.0f} | exp={best_exp} | DTE={dte} "
                f"| mid=${mid_price:.2f} | IV={iv:.1%} | OI={oi}"
            )

            return {
                "occ_symbol": occ_symbol,
                "underlying": symbol,
                "option_type": option_type,
                "strike": strike,
                "expiration": best_exp,
                "dte": dte,
                "mid_price": mid_price,
                "bid": float(best.get("bid", mid_price)),
                "ask": float(best.get("ask", mid_price)),
                "implied_volatility": iv,
                "open_interest": oi,
                "volume": vol,
                "premium_per_contract": round(mid_price * 100, 2),  # 1 contract = 100 shares
            }

        except Exception as e:
            logger.error(f"{symbol}: Contract selection failed: {e}")
            return None

    def find_spread_contracts(
        self,
        symbol: str,
        spread_type: str,   # "BULL_SPREAD" (call spread) or "BEAR_SPREAD" (put spread)
        target_dte: int = 35,
    ) -> Optional[Dict]:
        """
        Find both legs of a vertical spread.
        Bull call spread: buy lower call, sell higher call
        Bear put spread: buy higher put, sell lower put
        Returns dict with long_leg and short_leg contracts.
        """
        is_call_spread = spread_type == "BULL_SPREAD"
        option_type = "CALL" if is_call_spread else "PUT"
        cfg_opts = config.options

        try:
            ticker = yf.Ticker(symbol)
            opt_dates = ticker.options
            if not opt_dates:
                return None

            valid_dates = [
                d for d in opt_dates
                if cfg_opts.min_dte <= (datetime.strptime(d, "%Y-%m-%d").date() - date.today()).days <= cfg_opts.max_dte
            ]
            if not valid_dates:
                return None

            target_exp_date = date.today() + timedelta(days=target_dte)
            best_exp = min(valid_dates, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - target_exp_date).days))

            chain = ticker.option_chain(best_exp)
            df = chain.calls if is_call_spread else chain.puts

            if df is None or df.empty:
                return None

            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            current_price = float(hist["Close"].iloc[-1])

            df = df.copy()
            df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
            if df.empty:
                return None
            df["mid"] = (df["bid"] + df["ask"]) / 2
            df["strike_dist"] = abs(df["strike"] - current_price)

            # Long leg: ATM (closest to current price)
            long_row = df.loc[df["strike_dist"].idxmin()]
            long_strike = float(long_row["strike"])

            # Short leg: 3% OTM from long leg
            if is_call_spread:
                short_strike_target = long_strike * (1 + cfg_opts.spread_width_pct)
                short_candidates = df[df["strike"] >= long_strike * 1.01]
            else:
                short_strike_target = long_strike * (1 - cfg_opts.spread_width_pct)
                short_candidates = df[df["strike"] <= long_strike * 0.99]

            if short_candidates.empty:
                return None

            short_candidates = short_candidates.copy()
            short_candidates["target_dist"] = abs(short_candidates["strike"] - short_strike_target)
            short_row = short_candidates.loc[short_candidates["target_dist"].idxmin()]
            short_strike = float(short_row["strike"])

            exp_date = datetime.strptime(best_exp, "%Y-%m-%d").date()
            dte = (exp_date - date.today()).days

            long_mid = float(long_row.get("mid", long_row.get("lastPrice", 0)))
            short_mid = float(short_row.get("mid", short_row.get("lastPrice", 0)))
            net_debit = round((long_mid - short_mid) * 100, 2)  # per spread (100 shares)

            if net_debit <= 0:
                logger.warning(f"{symbol}: Spread has zero or negative debit — skipping")
                return None

            max_profit = round((short_strike - long_strike) * 100 - net_debit, 2) if is_call_spread else \
                         round((long_strike - short_strike) * 100 - net_debit, 2)

            return {
                "spread_type": spread_type,
                "underlying": symbol,
                "option_type": option_type,
                "expiration": best_exp,
                "dte": dte,
                "long_leg": {
                    "occ_symbol": long_row.get("contractSymbol", _build_occ_symbol(symbol, exp_date, option_type, long_strike)),
                    "strike": long_strike,
                    "mid": long_mid,
                    "side": "BUY",
                },
                "short_leg": {
                    "occ_symbol": short_row.get("contractSymbol", _build_occ_symbol(symbol, exp_date, option_type, short_strike)),
                    "strike": short_strike,
                    "mid": short_mid,
                    "side": "SELL",
                },
                "net_debit_per_spread": net_debit,
                "max_profit_per_spread": max_profit,
                "max_loss_per_spread": net_debit,
            }

        except Exception as e:
            logger.error(f"{symbol}: Spread selection failed: {e}")
            return None

    # ── Credit Spread Selection (sell premium, defined-risk) ─────────

    def _fetch_chain_rows(self, symbol: str, option_type: str, target_dte: int):
        """
        Fetch liquid option rows (via Alpaca data API) for the single expiration closest
        to target_dte. Returns (rows, current_price, exp_date_obj) or (None, None, None).
        Each row: {occ, strike, dte, mid, bid, ask, iv}.
        """
        try:
            from alpaca.data.historical import OptionHistoricalDataClient
            from alpaca.data.requests import OptionChainRequest
        except ImportError:
            return None, None, None
        try:
            cfg_opts = config.options
            data_client = OptionHistoricalDataClient(
                api_key=config.alpaca.api_key, secret_key=config.alpaca.secret_key,
            )
            hist = yf.Ticker(symbol).history(period="2d")
            if hist.empty:
                return None, None, None
            current_price = float(hist["Close"].iloc[-1])

            min_exp = date.today() + timedelta(days=cfg_opts.min_dte)
            max_exp = date.today() + timedelta(days=cfg_opts.max_dte)
            # Wider strike window than directional buys — credit spreads reach further OTM.
            request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                type=option_type.lower(),
                strike_price_gte=str(round(current_price * 0.70, 2)),
                strike_price_lte=str(round(current_price * 1.30, 2)),
            )
            snapshots = data_client.get_option_chain(request)
            if not snapshots:
                return None, None, None

            by_exp: Dict[date, list] = {}
            for occ_symbol, snap in snapshots.items():
                q = snap.latest_quote
                if not q or not q.bid_price or not q.ask_price:
                    continue
                bid, ask = float(q.bid_price), float(q.ask_price)
                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2
                if mid <= 0 or (ask - bid) / mid > cfg_opts.max_bid_ask_spread_pct:
                    continue
                try:
                    strike = int(occ_symbol[-8:]) / 1000
                    exp_obj = datetime.strptime(occ_symbol[-15:-9], "%y%m%d").date()
                    dte = (exp_obj - date.today()).days
                except Exception:
                    continue
                if not (cfg_opts.min_dte <= dte <= cfg_opts.max_dte):
                    continue
                by_exp.setdefault(exp_obj, []).append({
                    "occ": occ_symbol, "strike": strike, "dte": dte,
                    "mid": round(mid, 2), "bid": bid, "ask": ask,
                    "iv": float(snap.implied_volatility) if snap.implied_volatility else 0.0,
                })
            if not by_exp:
                return None, None, None

            target_exp_date = date.today() + timedelta(days=target_dte)
            best_exp = min(by_exp.keys(), key=lambda d: abs((d - target_exp_date).days))
            return by_exp[best_exp], current_price, best_exp
        except Exception as e:
            logger.debug(f"{symbol}: credit-spread chain fetch failed ({e})")
            return None, None, None

    def find_credit_spread_contracts(
        self,
        symbol: str,
        spread_kind: str,   # "SELL_PUT_SPREAD" (bullish) or "SELL_CALL_SPREAD" (bearish)
        target_dte: int = 35,
    ) -> Optional[Dict]:
        """
        Find both legs of a defined-risk credit spread.
          • Bull put spread (SELL_PUT_SPREAD): SELL a put ~credit_short_delta OTM (below price),
            BUY a further-OTM (lower) put as the cap. Profits if the stock stays above the short.
          • Bear call spread (SELL_CALL_SPREAD): SELL a call ~credit_short_delta OTM (above price),
            BUY a further-OTM (higher) call as the cap. Profits if the stock stays below the short.
        Short strike chosen by Black-Scholes delta (≈ credit_short_delta); falls back to a
        %-OTM proxy if greeks/IV are unavailable. Returns a spread dict or None.
        """
        cfg_opts = config.options
        is_put = spread_kind == "SELL_PUT_SPREAD"
        option_type = "PUT" if is_put else "CALL"

        rows, price, exp_obj = self._fetch_chain_rows(symbol, option_type, target_dte)
        if not rows or not price or not exp_obj:
            logger.debug(f"{symbol}: no chain rows for {spread_kind}")
            return None

        dte = (exp_obj - date.today()).days
        T = max(dte, 1) / 365.0
        RFR = 0.045
        for r in rows:
            r["abs_delta"] = abs(_bs_delta(price, r["strike"], T, RFR, r["iv"], option_type.lower())) if r["iv"] > 0 else 0.0

        # Only OTM strikes are sold (puts below price / calls above price)
        otm = [r for r in rows if (r["strike"] < price if is_put else r["strike"] > price)]
        if not otm:
            return None

        # Short leg: closest to the target delta; fall back to a %-OTM proxy without greeks
        with_delta = [r for r in otm if r["abs_delta"] > 0]
        if with_delta:
            short = min(with_delta, key=lambda r: abs(r["abs_delta"] - cfg_opts.credit_short_delta))
        else:
            proxy = price * (1 - cfg_opts.credit_spread_width_pct) if is_put else price * (1 + cfg_opts.credit_spread_width_pct)
            short = min(otm, key=lambda r: abs(r["strike"] - proxy))
        short_strike = short["strike"]

        # Long leg: ~width_pct of price further OTM than the short (the protective cap)
        width_pts = max(price * cfg_opts.credit_spread_width_pct, 0.01)
        if is_put:
            long_target = short_strike - width_pts
            longs = [r for r in otm if r["strike"] < short_strike]
        else:
            long_target = short_strike + width_pts
            longs = [r for r in otm if r["strike"] > short_strike]
        if not longs:
            return None
        long = min(longs, key=lambda r: abs(r["strike"] - long_target))
        long_strike = long["strike"]

        width = abs(short_strike - long_strike)
        credit_per_share = short["mid"] - long["mid"]
        if width <= 0 or credit_per_share <= 0:
            logger.debug(f"{symbol}: degenerate {spread_kind} (width={width}, credit={credit_per_share})")
            return None

        credit_per_spread = round(credit_per_share * 100, 2)
        width_per_spread = round(width * 100, 2)
        result = {
            "strategy_type": spread_kind,
            "underlying": symbol,
            "option_type": option_type,
            "expiration": exp_obj.strftime("%Y-%m-%d"),
            "dte": dte,
            "short_leg": {"occ_symbol": short["occ"], "strike": short_strike, "mid": short["mid"],
                          "bid": short["bid"], "ask": short["ask"], "abs_delta": round(short["abs_delta"], 3), "side": "SELL"},
            "long_leg": {"occ_symbol": long["occ"], "strike": long_strike, "mid": long["mid"],
                         "bid": long["bid"], "ask": long["ask"], "side": "BUY"},
            "credit_per_share": round(credit_per_share, 2),
            "credit_per_spread": credit_per_spread,
            "width_points": round(width, 2),
            "width_per_spread": width_per_spread,
            "max_loss_per_spread": round(width_per_spread - credit_per_spread, 2),
            "credit_to_width": round(credit_per_spread / width_per_spread, 3) if width_per_spread else 0.0,
        }
        logger.info(
            f"{symbol}: {spread_kind} — SELL {short_strike:g}{option_type[0]}/BUY {long_strike:g}{option_type[0]} "
            f"exp {result['expiration']} | credit=${credit_per_spread:.0f} width=${width_per_spread:.0f} "
            f"(c/w={result['credit_to_width']:.0%}, shortΔ≈{short['abs_delta']:.2f})"
        )
        return result

    def find_iron_condor_contracts(self, symbol: str, target_dte: int = 35) -> Optional[Dict]:
        """Combine a bull put spread + a bear call spread (sell both sides). Range-bound play."""
        put_side = self.find_credit_spread_contracts(symbol, "SELL_PUT_SPREAD", target_dte)
        call_side = self.find_credit_spread_contracts(symbol, "SELL_CALL_SPREAD", target_dte)
        if not put_side or not call_side:
            return None
        credit_per_spread = round(put_side["credit_per_spread"] + call_side["credit_per_spread"], 2)
        # Iron-condor max loss = the wider single spread's width − total credit (only one side can lose)
        width_per_spread = max(put_side["width_per_spread"], call_side["width_per_spread"])
        return {
            "strategy_type": "IRON_CONDOR",
            "underlying": symbol,
            "option_type": "BOTH",
            "expiration": put_side["expiration"],
            "dte": put_side["dte"],
            "put_spread": put_side,
            "call_spread": call_side,
            "credit_per_spread": credit_per_spread,
            "width_per_spread": width_per_spread,
            "max_loss_per_spread": round(width_per_spread - credit_per_spread, 2),
            "credit_to_width": round(credit_per_spread / width_per_spread, 3) if width_per_spread else 0.0,
        }

    # ── Order Placement ──────────────────────────────────────────────

    def buy_option(self, contract: Dict, contracts: int = 1, reason: str = "") -> Optional[Dict]:
        """
        Place a market BUY order for an option contract.
        Waits up to 10s for the actual fill price from Alpaca — that becomes
        premium_paid in the DB, not the pre-order mid price estimate.
        """
        occ_symbol = contract["occ_symbol"]
        try:
            # Cap the price we'll pay at ask × (1 + buffer) via a marketable limit order.
            # A plain market order on a wide/illiquid spread can fill far above the quote;
            # this bounds the worst-case fill while still crossing the spread to fill now.
            ask = float(contract.get("ask") or 0)
            if ask > 0:
                limit_price = round(ask * (1 + config.options.max_fill_slippage_pct), 2)
                order_request = LimitOrderRequest(
                    symbol=occ_symbol,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            order = self.client.submit_order(order_request)

            # Poll for actual fill price (paper orders fill in <1s, give 10s buffer)
            filled_price = None
            for _ in range(10):
                time.sleep(1)
                try:
                    refreshed = self.client.get_order_by_id(str(order.id))
                    if refreshed.filled_avg_price and float(refreshed.filled_avg_price) > 0:
                        filled_price = float(refreshed.filled_avg_price)
                        break
                except Exception:
                    pass

            # Use actual fill price; fall back to mid estimate if poll timed out
            actual_price_per_share = filled_price if filled_price else contract["mid_price"]
            actual_premium_per_contract = round(actual_price_per_share * 100, 2)
            actual_total_cost = round(actual_premium_per_contract * contracts, 2)

            logger.info(
                f"BUY OPTION: {occ_symbol} x{contracts} "
                f"| fill=${actual_price_per_share:.2f}/share "
                f"| total=${actual_total_cost:.0f} | {reason}"
            )
            return {
                "order_id": str(order.id),
                "occ_symbol": occ_symbol,
                "underlying": contract["underlying"],
                "option_type": contract["option_type"],
                "strike": contract["strike"],
                "expiration": contract["expiration"],
                "dte": contract["dte"],
                "contracts": contracts,
                "premium_per_contract": actual_premium_per_contract,
                "total_cost": actual_total_cost,
                "implied_volatility": contract.get("implied_volatility"),
                "filled_avg_price": actual_price_per_share,
                "side": "buy",
                "status": str(order.status),
            }
        except Exception as e:
            logger.error(f"BUY OPTION failed for {occ_symbol}: {e}")
            return None

    def buy_spread(self, spread: Dict, contracts: int = 1, reason: str = "") -> Optional[Dict]:
        """
        Place a vertical spread by submitting two separate orders.
        Long leg first, then short leg. Both market orders.
        """
        long_leg = spread["long_leg"]
        short_leg = spread["short_leg"]

        long_order = None
        short_order = None

        try:
            long_order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=long_leg["occ_symbol"],
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info(f"✅ SPREAD long leg: BUY {long_leg['occ_symbol']} x{contracts} | {reason}")
        except Exception as e:
            logger.error(f"❌ Spread long leg BUY failed ({long_leg['occ_symbol']}): {e}")
            return None

        try:
            short_order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=short_leg["occ_symbol"],
                    qty=contracts,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info(f"✅ SPREAD short leg: SELL {short_leg['occ_symbol']} x{contracts}")
        except Exception as e:
            logger.error(f"❌ Spread short leg SELL failed ({short_leg['occ_symbol']}): {e}")
            # Long leg filled, short leg failed — log clearly but don't cancel
            logger.critical(
                f"⚠️ UNCOVERED LONG: {long_leg['occ_symbol']} — short leg failed. "
                "Manual intervention may be required."
            )

        return {
            "spread_type": spread["spread_type"],
            "underlying": spread["underlying"],
            "long_occ": long_leg["occ_symbol"],
            "short_occ": short_leg["occ_symbol"] if short_order else None,
            "contracts": contracts,
            "net_debit": spread["net_debit_per_spread"] * contracts,
            "long_order_id": str(long_order.id) if long_order else None,
            "short_order_id": str(short_order.id) if short_order else None,
            "expiration": spread["expiration"],
            "dte": spread["dte"],
        }

    def sell_credit_spread(self, spread: Dict, contracts: int = 1, reason: str = "") -> Optional[Dict]:
        """
        Open a defined-risk credit spread as a SINGLE atomic multi-leg (MLEG) order.
        Submitting both legs in one order guarantees we never end up with a naked short
        leg (the failure mode the old two-order buy_spread warns about) and satisfies
        Alpaca's rule that all legs be covered within the same MLEG order.
        Records the ACTUAL net credit from the filled legs.
        """
        kind = spread["strategy_type"]
        legs_def = []  # (occ_symbol, OrderSide)
        if kind == "IRON_CONDOR":
            for side_spread in (spread["put_spread"], spread["call_spread"]):
                legs_def.append((side_spread["short_leg"]["occ_symbol"], OrderSide.SELL))
                legs_def.append((side_spread["long_leg"]["occ_symbol"], OrderSide.BUY))
            primary_short = spread["put_spread"]["short_leg"]["occ_symbol"]
            primary_long = spread["put_spread"]["long_leg"]["occ_symbol"]
        else:
            legs_def.append((spread["short_leg"]["occ_symbol"], OrderSide.SELL))
            legs_def.append((spread["long_leg"]["occ_symbol"], OrderSide.BUY))
            primary_short = spread["short_leg"]["occ_symbol"]
            primary_long = spread["long_leg"]["occ_symbol"]

        legs = [OptionLegRequest(symbol=occ, side=side, ratio_qty=1) for occ, side in legs_def]

        def _filled_credit(o):
            """Net credit per spread ($) once ALL legs are filled, else None."""
            fl = getattr(o, "legs", None) or []
            if fl and all(getattr(l, "filled_avg_price", None) for l in fl):
                net = 0.0
                for l in fl:
                    fp = float(l.filled_avg_price)
                    net += fp if str(l.side).lower().endswith("sell") else -fp  # collect on sells, pay on buys
                return round(net * 100, 2)
            return None

        # Price the spread as a marketable LIMIT order off the mid — NEVER a market order.
        # A market MLEG crosses both legs' bid-ask and historically surrendered 25–54% of the
        # mid credit (2026-06-17). We accept a net credit no worse than mid × (1 − buffer); if
        # the book won't meet it we cancel and SKIP rather than fill at a runaway price.
        mid_credit_per_share = spread["credit_per_spread"] / 100.0
        limit_price = round(mid_credit_per_share * (1 - config.options.spread_entry_slippage_pct), 2)
        if limit_price <= 0:
            logger.info(
                f"{spread['underlying']}: {kind} mid credit ${mid_credit_per_share:.2f}/sh too thin "
                f"to price a safe limit — skipping"
            )
            return None
        try:
            order = self.client.submit_order(
                LimitOrderRequest(
                    qty=contracts,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    legs=legs,
                    limit_price=limit_price,
                )
            )
        except Exception as e:
            logger.error(f"❌ SELL SPREAD failed for {spread['underlying']} {kind}: {e}")
            return None

        order_id = str(order.id)
        # Poll for the real fill. Don't fall back to the quote — a credit we didn't actually
        # collect is what produced the too-tight stops.
        actual_credit_per_spread = None
        waited = 0
        while waited < config.options.spread_fill_wait_seconds:
            time.sleep(1)
            waited += 1
            try:
                refreshed = self.client.get_order_by_id(order_id)
            except Exception:
                continue
            actual_credit_per_spread = _filled_credit(refreshed)
            if actual_credit_per_spread is not None:
                break
            if str(getattr(refreshed, "status", "")).lower() in ("canceled", "rejected", "expired"):
                break

        if actual_credit_per_spread is None:
            # Limit not met in time — cancel and skip. Then RE-CHECK after cancelling: a fill
            # can land in the same instant, and silently dropping it would leave an untracked
            # (orphan) position open in Alpaca. If it did fill, keep it and book the real credit.
            try:
                self.client.cancel_order_by_id(order_id)
            except Exception:
                pass
            try:
                time.sleep(1)
                actual_credit_per_spread = _filled_credit(self.client.get_order_by_id(order_id))
            except Exception:
                actual_credit_per_spread = None
            if actual_credit_per_spread is None:
                logger.info(
                    f"{spread['underlying']}: {kind} did not fill at limit ${limit_price:.2f}/sh "
                    f"(≥ {1 - config.options.spread_entry_slippage_pct:.0%} of ${mid_credit_per_share:.2f} mid) "
                    f"in {config.options.spread_fill_wait_seconds}s — skipping to avoid a bad fill"
                )
                return None

        total_credit = round(actual_credit_per_spread * contracts, 2)
        max_loss_per_spread = round(spread["width_per_spread"] - actual_credit_per_spread, 2)
        logger.info(
            f"✅ SELL SPREAD: {spread['underlying']} {kind} x{contracts} "
            f"| credit=${total_credit:.0f} | max loss=${max_loss_per_spread * contracts:.0f} | {reason}"
        )
        return {
            "order_id": order_id,
            "strategy_type": kind,
            "underlying": spread["underlying"],
            "short_occ": primary_short,
            "long_occ": primary_long,
            "contracts": contracts,
            "credit_per_spread": actual_credit_per_spread,
            "total_credit": total_credit,
            "width_per_spread": spread["width_per_spread"],
            "max_loss_per_spread": max_loss_per_spread,
            "expiration": spread["expiration"],
            "dte": spread["dte"],
            "status": str(order.status),
        }

    def close_credit_spread(self, short_occ: str, long_occ: str, contracts: int,
                            reason: str = "", limit_debit_per_share: Optional[float] = None,
                            allow_market_fallback: bool = False) -> Optional[Dict]:
        """
        Close a credit spread as a SINGLE atomic multi-leg (MLEG) order — buy back the
        short leg and sell the long leg together. Doing both in one order means we are
        never momentarily naked, which is what caused the "uncovered" rejection that
        stranded the NVDA long leg on 2026-06-12 (the old two-order close_spread).

        When `limit_debit_per_share` is given the buy-to-close is a marketable LIMIT order:
        we pay no more than that net debit per share. If it doesn't fill in time, a STOP
        (allow_market_fallback=True) escalates to a market order — we MUST get out — while a
        profit-take returns None so the caller keeps the trade open and retries next cycle.
        With no limit (orphan cleanup) it submits a market order directly.

        Polls for fills and returns the ACTUAL cost-to-close from the filled legs so P&L is
        booked off real prices, not trigger-moment marks. `actual_cost_to_close` is None if
        fills couldn't be confirmed (caller falls back to the mark).
        """
        legs = [
            OptionLegRequest(symbol=short_occ, side=OrderSide.BUY, ratio_qty=1),   # buy back short
            OptionLegRequest(symbol=long_occ, side=OrderSide.SELL, ratio_qty=1),   # sell the long
        ]

        def _filled_cost(o):
            """Net cost-to-close ($, all contracts) once ALL legs are filled, else None."""
            fl = getattr(o, "legs", None) or []
            if fl and all(getattr(l, "filled_avg_price", None) for l in fl):
                net = 0.0
                for l in fl:
                    fp = float(l.filled_avg_price)
                    # BUY (closing the short) costs us; SELL (closing the long) pays us back.
                    net += fp if str(l.side).lower().endswith("buy") else -fp
                return round(net * 100 * contracts, 2)
            return None

        def _submit(use_limit):
            """Submit one close order. Returns (order, special): `special` is a dict for the
            already-closed / PDT cases (order is None then); on hard failure both are None."""
            try:
                if use_limit:
                    req = LimitOrderRequest(
                        qty=contracts, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                        legs=legs, limit_price=round(limit_debit_per_share, 2),
                    )
                else:
                    req = MarketOrderRequest(
                        qty=contracts, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                        legs=legs,
                    )
                return self.client.submit_order(req), None
            except Exception as e:
                err = str(e)
                if "40410000" in err or "position not found" in err.lower() or "no open position" in err.lower():
                    logger.warning(f"CLOSE SPREAD: {short_occ}/{long_occ} already gone — treating as closed | {e}")
                    return None, {"already_closed": True, "short_occ": short_occ, "long_occ": long_occ}
                if "40310100" in err or "pattern day trading" in err.lower():
                    logger.warning(f"CLOSE SPREAD PDT-blocked: {short_occ}/{long_occ} — retry next day | {e}")
                    return None, {"pdt_blocked": True, "short_occ": short_occ, "long_occ": long_occ}
                logger.error(f"❌ CLOSE SPREAD failed for {short_occ}/{long_occ}: {e}")
                return None, None

        def _await_fill(order_id):
            waited = 0
            while waited < config.options.spread_fill_wait_seconds:
                time.sleep(1)
                waited += 1
                try:
                    refreshed = self.client.get_order_by_id(order_id)
                except Exception:
                    continue
                cost = _filled_cost(refreshed)
                if cost is not None:
                    return cost
                if str(getattr(refreshed, "status", "")).lower() in ("canceled", "rejected", "expired"):
                    return None
            return None

        use_limit = limit_debit_per_share is not None and limit_debit_per_share > 0
        order, special = _submit(use_limit)
        if special is not None:
            return special
        if order is None:
            return None

        order_id = str(order.id)
        actual_cost_to_close = _await_fill(order_id)

        if actual_cost_to_close is None and use_limit:
            # Limit didn't fill: cancel, re-check for a same-instant fill (orphan guard), then
            # escalate to market on a STOP (must exit) or retry next cycle on a profit-take.
            try:
                self.client.cancel_order_by_id(order_id)
            except Exception:
                pass
            try:
                actual_cost_to_close = _filled_cost(self.client.get_order_by_id(order_id))
            except Exception:
                actual_cost_to_close = None
            if actual_cost_to_close is None:
                if allow_market_fallback:
                    logger.info(f"CLOSE SPREAD: {short_occ}/{long_occ} limit unfilled — escalating to market (stop)")
                    order, special = _submit(False)
                    if special is not None:
                        return special
                    if order is None:
                        return None
                    order_id = str(order.id)
                    actual_cost_to_close = _await_fill(order_id)
                else:
                    logger.info(f"CLOSE SPREAD: {short_occ}/{long_occ} limit unfilled (profit-take) — retry next cycle")
                    return None

        logger.info(f"✅ CLOSE SPREAD (atomic): BUY {short_occ} / SELL {long_occ} x{contracts} | {reason}")
        return {
            "order_id": order_id,
            "short_occ": short_occ,
            "long_occ": long_occ,
            "contracts": contracts,
            "actual_cost_to_close": actual_cost_to_close,
            "status": "filled" if actual_cost_to_close is not None else "submitted",
        }

    def close_option(self, occ_symbol: str, contracts: int = None, reason: str = "") -> Optional[Dict]:
        """
        Close an open option position (sell to close).
        If contracts=None, closes the full position.
        """
        try:
            if contracts is None:
                self.client.close_position(occ_symbol)
                logger.info(f"✅ CLOSE OPTION (full): {occ_symbol} | {reason}")
                return {"occ_symbol": occ_symbol, "contracts": "all", "side": "sell", "reason": reason}
            else:
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=occ_symbol,
                        qty=contracts,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                logger.info(f"✅ CLOSE OPTION: {occ_symbol} x{contracts} | {reason}")
                return {
                    "order_id": str(order.id),
                    "occ_symbol": occ_symbol,
                    "contracts": contracts,
                    "side": "sell",
                    "reason": reason,
                    "status": str(order.status),
                }
        except Exception as e:
            err_str = str(e)
            if "40410000" in err_str or "position not found" in err_str.lower() or "no open position" in err_str.lower():
                logger.warning(f"CLOSE OPTION: {occ_symbol} already gone at Alpaca — marking closed | {e}")
                return {"occ_symbol": occ_symbol, "contracts": contracts or "all", "side": "sell", "reason": reason, "already_closed": True}
            if "40310100" in err_str or "pattern day trading" in err_str.lower():
                logger.warning(f"CLOSE OPTION PDT-blocked: {occ_symbol} — bought and sold same day, will retry tomorrow | {e}")
                return {"occ_symbol": occ_symbol, "contracts": contracts or "all", "side": "sell", "reason": reason, "pdt_blocked": True}
            logger.error(f"❌ CLOSE OPTION failed for {occ_symbol}: {e}")
            return None

    def close_spread(self, long_occ: str, short_occ: str, contracts: int = None, reason: str = "") -> Optional[Dict]:
        """
        Close a spread position: buy to cover the short leg FIRST, then sell the long leg.
        Order matters — selling long first while short is open causes an uncovered rejection.
        """
        results: Dict = {}

        # Step 1: Buy to cover short leg
        try:
            short_pos = self.client.get_open_position(short_occ)
            short_qty = abs(int(float(short_pos.qty))) if contracts is None else contracts
            order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=short_occ,
                    qty=short_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info(f"✅ SPREAD close short leg: BUY {short_occ} x{short_qty} | {reason}")
            results["short_close_order_id"] = str(order.id)
        except Exception as e:
            logger.warning(f"Short leg {short_occ} already gone or close failed ({e}) — proceeding to long")

        # Step 2: Sell to close long leg
        long_result = self.close_option(long_occ, contracts, reason=f"spread long close | {reason}")
        if long_result:
            results["long_close"] = long_result

        return results if results else None

    def cancel_all_orders(self):
        try:
            cancelled = self.client.cancel_orders()
            logger.info(f"Cancelled {len(cancelled)} open orders")
        except Exception as e:
            logger.warning(f"Cancel orders failed: {e}")

    def get_recent_orders(self, limit: int = 20) -> List[Dict]:
        try:
            orders = self.client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            )
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else None,
                    "side": str(o.side),
                    "status": str(o.status),
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "submitted_at": str(o.submitted_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch recent orders: {e}")
            return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_occ_symbol(underlying: str, exp_date: date, option_type: str, strike: float) -> str:
    """
    Build OCC option symbol manually as fallback.
    Format: AAPL240119C00150000
    """
    root = underlying.upper().ljust(6)[:6].rstrip()
    date_str = exp_date.strftime("%y%m%d")
    cp = "C" if option_type == "CALL" else "P"
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{root}{date_str}{cp}{strike_str}"
