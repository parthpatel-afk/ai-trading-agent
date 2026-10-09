"""
AI Investor v2 — Options Executor
Option contract selection + order placement for the EMA strategy.

Contract rule: out-of-the-money by `otm_strikes` increments, on the standard MONTHLY
expiration (3rd Friday) closest to ~60 DTE (tighter spreads), liquidity-filtered.

Contract data: Alpaca options API primary, yfinance fallback.
Order execution: Alpaca Trading API (options via OCC symbol).

OCC symbol format: AAPL240119C00150000
  = underlying + YYMMDD + C/P + 8-digit strike (strike × 1000, zero-padded)
"""
import time
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict
import numpy as np
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from loguru import logger

from agent.config import config
from agent import pricing


def _otm_target_strike(strikes: List[float], current_price: float, option_type: str, n: int) -> Optional[float]:
    """
    Pick the strike `n` increments out-of-the-money from the at-the-money strike.
    Calls → n strikes above spot; puts → n strikes below. Clamped to the ladder.
    """
    uniq = sorted(set(round(float(s), 4) for s in strikes))
    if not uniq:
        return None
    atm_i = min(range(len(uniq)), key=lambda i: abs(uniq[i] - current_price))
    if option_type == "CALL":
        idx = min(atm_i + n, len(uniq) - 1)
    else:
        idx = max(atm_i - n, 0)
    return uniq[idx]


def _third_friday(year: int, month: int) -> date:
    """The standard monthly options expiration: the 3rd Friday of the month."""
    d = date(year, month, 1)
    # weekday(): Mon=0 … Fri=4. Days until the first Friday, then +14 for the third.
    first_friday = 1 + ((4 - d.weekday()) % 7)
    return date(year, month, first_friday + 14)


def _target_monthly_date(target_dte: int, lo_dte: int, hi_dte: int, today: Optional[date] = None) -> Optional[date]:
    """
    Pick the standard MONTHLY expiration (3rd Friday) whose DTE is closest to `target_dte`,
    considering only monthlies whose DTE falls in [lo_dte, hi_dte]. Returns None if none fit
    (caller then falls back to nearest-available behaviour).
    """
    today = today or date.today()
    candidates = []
    y, m = today.year, today.month
    for _ in range(6):  # scan the next ~6 monthly expirations
        exp = _third_friday(y, m)
        dte = (exp - today).days
        if lo_dte <= dte <= hi_dte:
            candidates.append((abs(dte - target_dte), exp))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _choose_expiration(available: List[date], today: Optional[date] = None) -> Optional[date]:
    """
    Choose which expiration to trade from the ones available in the chain.
    Prefers the standard MONTHLY (3rd Friday) closest to target DTE; falls back to the
    nearest available expiration to target if no monthly is present.
    """
    today = today or date.today()
    cfg = config.options
    if not available:
        return None
    if cfg.prefer_monthly:
        tgt = _target_monthly_date(cfg.target_dte, cfg.min_dte, cfg.max_dte, today)
        if tgt and tgt in available:
            return tgt
        monthlies = [d for d in available if d == _third_friday(d.year, d.month)]
        if monthlies:
            return min(monthlies, key=lambda d: abs((d - today).days - cfg.target_dte))
    return min(available, key=lambda d: abs((d - today).days - cfg.target_dte))


def _norm_cdf(x: float) -> float:
    return pricing.norm_cdf(x)


def _bs_delta(spot: float, strike: float, dte: int, iv: float, option_type: str, r: float = 0.04) -> Optional[float]:
    """Black-Scholes delta MAGNITUDE (0..1). Thin wrapper over the shared pricer
    (agent.pricing) so live selection and the backtester share ONE formula."""
    return pricing.bs_delta(spot, strike, dte, iv, option_type, r)


def _select_by_delta(rows, spot, option_type, target_delta, delta_min, delta_max, r):
    """Pick the contract whose |delta| is closest to target, preferring the
    [delta_min, delta_max] band, constrained to the IN-THE-MONEY side
    (calls: strike <= spot; puts: strike >= spot). Each row needs strike/dte/iv.
    Falls back to a moneyness proxy when IV (hence delta) is missing.
    Returns (row, delta_or_None) or (None, None)."""
    if option_type == "CALL":
        sided = [x for x in rows if x["strike"] <= spot] or rows
    else:
        sided = [x for x in rows if x["strike"] >= spot] or rows

    scored = []
    for x in sided:
        d = _bs_delta(spot, x["strike"], x.get("dte", 60), x.get("iv", 0) or 0, option_type, r)
        if d is not None:
            scored.append((x, d))
    if scored:
        in_band = [(x, d) for x, d in scored if delta_min <= d <= delta_max]
        pool = in_band or scored
        x, d = min(pool, key=lambda t: abs(t[1] - target_delta))
        return x, round(d, 3)

    # Fallback: no usable IV → approximate depth by moneyness (deeper target → deeper strike).
    itm_pct = 0.05 + (target_delta - 0.70) * 0.20
    proxy = spot * (1 - itm_pct) if option_type == "CALL" else spot * (1 + itm_pct)
    x = min(sided, key=lambda r2: abs(r2["strike"] - proxy))
    return x, None


class Executor:
    """Wraps Alpaca for options order placement; selects contracts before ordering."""

    def __init__(self):
        self.client = TradingClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            paper=config.alpaca.paper,
        )
        self._opt_data_client = None
        self._opt_quote_cache: Dict[str, tuple] = {}   # occ_symbol -> (ts, {bid,ask,mid})
        mode = "PAPER" if config.alpaca.paper else "LIVE 💰"
        logger.info(f"Executor initialized — Mode: {mode}")

    def get_option_quote(self, occ_symbol: str, ttl: int = 120) -> Optional[Dict]:
        """
        Latest bid/ask/mid for an option contract (used to show a fair MID price on the
        dashboard instead of the scary bid-only mark — review item 8). Cached `ttl` seconds
        so frequent dashboard polls don't spam the data API.
        """
        now = time.time()
        hit = self._opt_quote_cache.get(occ_symbol)
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            if self._opt_data_client is None:
                from alpaca.data.historical import OptionHistoricalDataClient
                self._opt_data_client = OptionHistoricalDataClient(
                    api_key=config.alpaca.api_key, secret_key=config.alpaca.secret_key)
            from alpaca.data.requests import OptionLatestQuoteRequest
            q = self._opt_data_client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol))
            qq = q[occ_symbol]
            bid, ask = float(qq.bid_price or 0), float(qq.ask_price or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (bid or ask or 0)
            out = {"bid": bid, "ask": ask, "mid": round(mid, 2)}
            self._opt_quote_cache[occ_symbol] = (now, out)
            return out
        except Exception as e:
            logger.debug(f"{occ_symbol}: option quote fetch failed: {e}")
            return None

    # ── Account & Positions ──────────────────────────────────────────

    def get_account(self) -> Dict:
        acct = self.client.get_account()
        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "status": str(acct.status),
            "trading_blocked": acct.trading_blocked,
            "paper": config.alpaca.paper,
        }

    def get_positions(self) -> List[Dict]:
        """All open positions (stocks + options)."""
        positions = self.client.get_all_positions()
        result = []
        for p in positions:
            # Normalize asset class to "us_option"/"us_equity". The enum stringifies as
            # "AssetClass.US_OPTION", so a naive == "us_option" check was always False —
            # which made num_positions show 0 and get_option_positions return nothing.
            ac_raw = getattr(p, "asset_class", None)
            ac_str = str(getattr(ac_raw, "value", ac_raw) or "").lower()
            asset_class = "us_option" if "option" in ac_str else "us_equity"
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": str(p.side),
                "asset_class": asset_class,
            })
        return result

    def get_option_positions(self) -> List[Dict]:
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
            return self.client.get_clock().is_open
        except Exception as e:
            logger.warning(f"Market clock check failed: {e}")
            return False

    # ── Option Contract Selection ────────────────────────────────────

    def _find_contract_via_alpaca(self, symbol: str, option_type: str, target_delta: Optional[float] = None) -> Optional[Dict]:
        """
        Find a DEEP-ITM contract (≈target_delta) via Alpaca's options data API (primary source).
        Returns None silently if unavailable so the caller falls back to yfinance.
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

            hist = yf.Ticker(symbol).history(period="2d")
            if hist.empty:
                return None
            current_price = float(hist["Close"].iloc[-1])

            min_exp = date.today() + timedelta(days=cfg_opts.min_dte)
            max_exp = date.today() + timedelta(days=cfg_opts.max_dte)

            request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                type=option_type.lower(),
                strike_price_gte=str(round(current_price * 0.80, 2)),
                strike_price_lte=str(round(current_price * 1.20, 2)),
            )
            snapshots = data_client.get_option_chain(request)
            if not snapshots:
                return None

            # Group quotable contracts by expiration.
            by_expiration: Dict[date, list] = {}
            for occ_symbol, snap in snapshots.items():
                q = snap.latest_quote
                if not q or not q.bid_price or not q.ask_price:
                    continue
                bid, ask = float(q.bid_price), float(q.ask_price)
                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2
                if mid <= 0:
                    continue
                try:
                    strike = int(occ_symbol[-8:]) / 1000
                    exp_obj = datetime.strptime(occ_symbol[-15:-9], "%y%m%d").date()
                    dte = (exp_obj - date.today()).days
                except Exception:
                    continue
                if not (cfg_opts.min_dte <= dte <= cfg_opts.max_dte):
                    continue
                by_expiration.setdefault(exp_obj, []).append({
                    "occ_symbol": occ_symbol, "strike": strike, "dte": dte, "mid": mid,
                    "bid": bid, "ask": ask, "spread_pct": (ask - bid) / mid,
                    "iv": float(snap.implied_volatility) if snap.implied_volatility else 0,
                    "exp_obj": exp_obj,
                })
            if not by_expiration:
                return None

            # Expiration: the standard monthly (3rd Friday) closest to ~60 DTE (1e).
            best_exp = _choose_expiration(list(by_expiration.keys()))
            if best_exp is None:
                return None
            rows = by_expiration[best_exp]

            # Pick the DEEP-ITM strike closest to the target delta (constrained to the ITM side).
            tdelta = target_delta if target_delta is not None else cfg_opts.target_delta
            best_row, picked_delta = _select_by_delta(
                rows, current_price, option_type, tdelta,
                cfg_opts.delta_min, cfg_opts.delta_max, cfg_opts.risk_free_rate)
            if best_row is None:
                return None

            best = {
                "occ_symbol": best_row["occ_symbol"],
                "underlying": symbol,
                "option_type": option_type,
                "strike": best_row["strike"],
                "expiration": best_row["exp_obj"].strftime("%Y-%m-%d"),
                "dte": best_row["dte"],
                "mid_price": round(best_row["mid"], 2),
                "bid": best_row["bid"],
                "ask": best_row["ask"],
                "implied_volatility": best_row["iv"],
                "delta": picked_delta,
                "open_interest": 0,
                "volume": 0,
                "premium_per_contract": round(best_row["mid"] * 100, 2),
            }
            logger.info(
                f"{symbol}: {option_type} via Alpaca {best['occ_symbol']} "
                f"| strike=${best['strike']:.2f} Δ≈{picked_delta if picked_delta is not None else 'n/a'} (target {tdelta:.2f}, deep-ITM) "
                f"| DTE={best['dte']} | mid=${best['mid_price']:.2f}"
            )
            return best
        except Exception as e:
            logger.debug(f"{symbol}: Alpaca options chain failed ({e})")
            return None

    def find_option_contract(self, symbol: str, option_type: str, target_delta: Optional[float] = None) -> Optional[Dict]:
        """
        Find the DEEP-ITM contract (≈target_delta, monthly ~target DTE). Alpaca first, yfinance fallback.
        Returns a contract dict or None.
        """
        result = self._find_contract_via_alpaca(symbol, option_type, target_delta)
        if result:
            return result
        logger.debug(f"{symbol}: Alpaca options unavailable — falling back to yfinance")

        cfg_opts = config.options
        tdelta = target_delta if target_delta is not None else cfg_opts.target_delta
        try:
            ticker = yf.Ticker(symbol)
            opt_dates = ticker.options
            if not opt_dates:
                logger.warning(f"{symbol}: No options dates available")
                return None

            valid_dates = [
                d for d in opt_dates
                if cfg_opts.min_dte <= (datetime.strptime(d, "%Y-%m-%d").date() - date.today()).days <= cfg_opts.max_dte
            ]
            if not valid_dates:
                logger.warning(f"{symbol}: No expirations in {cfg_opts.min_dte}-{cfg_opts.max_dte} DTE range")
                return None

            # Prefer the standard monthly (3rd Friday) closest to ~60 DTE (1e).
            best_exp_date = _choose_expiration([datetime.strptime(d, "%Y-%m-%d").date() for d in valid_dates])
            if best_exp_date is None:
                return None
            best_exp = best_exp_date.strftime("%Y-%m-%d")

            chain = ticker.option_chain(best_exp)
            df = chain.calls if option_type == "CALL" else chain.puts
            if df is None or df.empty:
                logger.warning(f"{symbol}: Empty {option_type} chain for {best_exp}")
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
            df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]

            exp_date = datetime.strptime(best_exp, "%Y-%m-%d").date()
            dte = (exp_date - date.today()).days

            # Liquidity filter (OI + volume + spread); relax to OI floor if needed.
            liquid = df[
                (df["openInterest"] >= cfg_opts.min_open_interest) &
                (df["volume"] >= cfg_opts.min_volume) &
                (df["spread_pct"] <= cfg_opts.max_bid_ask_spread_pct)
            ].copy()
            if liquid.empty:
                logger.debug(f"{symbol}: Relaxing liquidity filters")
                liquid = df[df["openInterest"] >= cfg_opts.min_open_interest_floor].copy()
                if liquid.empty:
                    logger.info(f"{symbol}: No contract meets OI floor ({cfg_opts.min_open_interest_floor}) — skipping")
                    return None

            # Pick the DEEP-ITM strike closest to the target delta.
            rows = [
                {"strike": float(r["strike"]),
                 "dte": dte,
                 "iv": float(r["impliedVolatility"]) if not np.isnan(r.get("impliedVolatility", float("nan"))) else 0,
                 "_row": r}
                for _, r in liquid.iterrows()
            ]
            picked, picked_delta = _select_by_delta(
                rows, current_price, option_type, tdelta,
                cfg_opts.delta_min, cfg_opts.delta_max, cfg_opts.risk_free_rate)
            if picked is None:
                return None
            best = picked["_row"]

            strike = float(best["strike"])
            mid_price = float(best.get("mid", best.get("lastPrice", 0)))
            if mid_price <= 0:
                return None

            occ_symbol = best.get("contractSymbol", "") or _build_occ_symbol(symbol, exp_date, option_type, strike)
            iv = float(best.get("impliedVolatility", 0)) if not np.isnan(best.get("impliedVolatility", float("nan"))) else 0
            oi_raw = best.get("openInterest", 0)
            oi = int(oi_raw) if oi_raw and not np.isnan(float(oi_raw)) else 0
            vol_raw = best.get("volume", 0)
            vol = int(vol_raw) if vol_raw and not np.isnan(float(vol_raw)) else 0

            logger.info(
                f"{symbol}: {option_type} via yfinance {occ_symbol} "
                f"| strike=${strike:.2f} Δ≈{picked_delta if picked_delta is not None else 'n/a'} (target {tdelta:.2f}, deep-ITM) | exp={best_exp} "
                f"| DTE={dte} | mid=${mid_price:.2f} | OI={oi}"
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
                "delta": picked_delta,
                "open_interest": oi,
                "volume": vol,
                "premium_per_contract": round(mid_price * 100, 2),
            }
        except Exception as e:
            logger.error(f"{symbol}: Contract selection failed: {e}")
            return None

    # ── Order Placement ──────────────────────────────────────────────

    def _wait_for_fill(self, order_id: str, wait_seconds: int):
        """Poll an order for up to `wait_seconds`, returning early once fully filled."""
        for _ in range(max(1, wait_seconds)):
            time.sleep(1)
            try:
                o = self.client.get_order_by_id(order_id)
                if str(o.status).lower().endswith("filled") and float(o.filled_qty or 0) > 0:
                    if "partial" not in str(o.status).lower():
                        break
            except Exception:
                pass

    def _order_fill(self, order_id: str):
        """Authoritative (filled_qty, filled_avg_price) for an order — read after cancelling."""
        try:
            o = self.client.get_order_by_id(order_id)
            return float(o.filled_qty or 0), float(o.filled_avg_price or 0)
        except Exception:
            return 0.0, 0.0

    def buy_option(self, contract: Dict, contracts: int = 1, reason: str = "") -> Optional[Dict]:
        """
        Place a BUY using a MID-priced limit "walk" (review item 1a):
        start the limit at the mid, step it up toward a ceiling of mid × (1 + ceiling_pct)
        capped at the ask, and SKIP the trade if it still won't fill at the ceiling — so a
        wide spread can't drag the fill up to the ask. Handles partial fills (keeps whatever
        filled). Returns None if nothing filled at a fair price (caller logs a skip).
        """
        occ_symbol = contract["occ_symbol"]
        cfg = config.options
        try:
            bid = float(contract.get("bid") or 0)
            ask = float(contract.get("ask") or 0)
            mid = float(contract.get("mid_price") or 0) or ((bid + ask) / 2 if bid > 0 and ask > 0 else 0)

            if mid <= 0 or ask <= 0:
                # No usable two-sided quote — fall back to a plain marketable order.
                order = self.client.submit_order(MarketOrderRequest(
                    symbol=occ_symbol, qty=contracts, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                self._wait_for_fill(str(order.id), 10)
                filled_qty, filled_avg = self._order_fill(str(order.id))
                if filled_qty <= 0:
                    logger.info(f"{occ_symbol}: market buy did not fill — skipping")
                    return None
                return self._buy_result(contract, order.id, filled_qty, filled_avg, "filled", reason)

            # Build the price ladder from mid → ceiling (ceiling = mid+ceiling_pct, never above ask).
            ceiling = min(round(mid * (1 + cfg.entry_mid_ceiling_pct), 2), round(ask, 2))
            ceiling = max(ceiling, round(mid, 2))
            steps = max(1, cfg.entry_walk_steps)
            ladder = []
            for k in range(steps + 1):
                p = round(mid + (ceiling - mid) * k / steps, 2)
                if p not in ladder:
                    ladder.append(p)

            remaining = contracts
            total_filled = 0.0
            cost_shares = 0.0   # sum(fill_price * filled_qty), for the weighted avg
            last_order_id = None

            for price in ladder:
                if remaining <= 0:
                    break
                order = self.client.submit_order(LimitOrderRequest(
                    symbol=occ_symbol, qty=remaining, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, limit_price=price))
                last_order_id = str(order.id)
                self._wait_for_fill(last_order_id, cfg.entry_walk_wait_seconds)
                # Cancel BEFORE reading the fill, so no further fills can sneak in after we
                # account for this step (avoids over-buying past `contracts`).
                self._safe_cancel(last_order_id)
                fq, favg = self._order_fill(last_order_id)
                if fq > 0:
                    total_filled += fq
                    cost_shares += favg * fq
                    remaining = contracts - int(total_filled)

            if total_filled <= 0:
                logger.info(
                    f"{occ_symbol}: SKIP — no fill walking mid ${mid:.2f} → ceiling ${ceiling:.2f} "
                    f"(ask ${ask:.2f}). Spread too wide to enter fairly. | {reason}")
                return None

            avg_price = cost_shares / total_filled
            if total_filled < contracts:
                logger.warning(f"{occ_symbol}: partial fill {int(total_filled)}/{contracts} @ ${avg_price:.2f}")
            return self._buy_result(contract, last_order_id, total_filled, avg_price, "filled", reason)
        except Exception as e:
            logger.error(f"BUY OPTION failed for {occ_symbol}: {e}")
            return None

    def _safe_cancel(self, order_id: str):
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception:
            pass

    def _buy_result(self, contract, order_id, filled_qty, avg_price, status, reason) -> Dict:
        contracts = int(filled_qty)
        per_contract = round(avg_price * 100, 2)
        total_cost = round(per_contract * contracts, 2)
        logger.info(
            f"BUY OPTION: {contract['occ_symbol']} x{contracts} "
            f"| fill=${avg_price:.2f}/share | total=${total_cost:.0f} | {reason}")
        return {
            "order_id": str(order_id),
            "occ_symbol": contract["occ_symbol"],
            "underlying": contract["underlying"],
            "option_type": contract["option_type"],
            "strike": contract["strike"],
            "expiration": contract["expiration"],
            "dte": contract["dte"],
            "contracts": contracts,
            "premium_per_contract": per_contract,
            "total_cost": total_cost,
            "implied_volatility": contract.get("implied_volatility"),
            "filled_avg_price": avg_price,
            "side": "buy",
            "status": str(status),
        }

    def close_option(self, occ_symbol: str, contracts: int = None, reason: str = "") -> Optional[Dict]:
        """Close (sell to close) an open option position. contracts=None closes the full position."""
        try:
            if contracts is None:
                self.client.close_position(occ_symbol)
                logger.info(f"✅ CLOSE OPTION (full): {occ_symbol} | {reason}")
                return {"occ_symbol": occ_symbol, "contracts": "all", "side": "sell", "reason": reason}
            order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=occ_symbol, qty=contracts, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info(f"✅ CLOSE OPTION: {occ_symbol} x{contracts} | {reason}")
            return {
                "order_id": str(order.id), "occ_symbol": occ_symbol, "contracts": contracts,
                "side": "sell", "reason": reason, "status": str(order.status),
            }
        except Exception as e:
            err_str = str(e)
            if "40410000" in err_str or "position not found" in err_str.lower() or "no open position" in err_str.lower():
                logger.warning(f"CLOSE OPTION: {occ_symbol} already gone at Alpaca — marking closed | {e}")
                return {"occ_symbol": occ_symbol, "contracts": contracts or "all", "side": "sell", "reason": reason, "already_closed": True}
            logger.error(f"❌ CLOSE OPTION failed for {occ_symbol}: {e}")
            return None

    def cancel_all_orders(self):
        try:
            cancelled = self.client.cancel_orders()
            logger.info(f"Cancelled {len(cancelled)} open orders")
        except Exception as e:
            logger.warning(f"Cancel orders failed: {e}")

    def get_recent_orders(self, limit: int = 20) -> List[Dict]:
        try:
            orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
            return [
                {
                    "id": str(o.id), "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else None,
                    "side": str(o.side), "status": str(o.status),
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
    """Build OCC option symbol. Format: AAPL240119C00150000"""
    root = underlying.upper().ljust(6)[:6].rstrip()
    date_str = exp_date.strftime("%y%m%d")
    cp = "C" if option_type == "CALL" else "P"
    strike_int = int(round(strike * 1000))
    return f"{root}{date_str}{cp}{strike_int:08d}"
