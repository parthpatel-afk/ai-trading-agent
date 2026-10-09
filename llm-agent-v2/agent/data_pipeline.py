"""
AI Investor Agent - Data Pipeline v2.0
Multi-source market data: Alpaca (real-time quotes + news) + yfinance (history).
"""
import time
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
import requests
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger

from agent.config import config

_ALPACA_DATA_TIMEOUT = 5  # seconds before falling back to yfinance


def _with_timeout(fn, *args, timeout=_ALPACA_DATA_TIMEOUT, **kwargs):
    """Run fn(*args, **kwargs) with a wall-clock timeout. Raises concurrent.futures.TimeoutError on hang."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result(timeout=timeout)


class DataPipeline:
    """
    Fetches and normalizes market data from all sources.
    Returns clean DataFrames ready for signal generation.
    """

    def __init__(self):
        self.alpaca_data = StockHistoricalDataClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )
        self._alpaca_news_headers = {
            "APCA-API-KEY-ID": config.alpaca.api_key,
            "APCA-API-SECRET-KEY": config.alpaca.secret_key,
        }
        logger.info("DataPipeline initialized")

    # ── Historical Price Data ────────────────────────────────────────

    def get_historical_bars(
        self,
        symbol: str,
        period_days: int = 65,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars via Alpaca, with yfinance fallback.
        65 days default gives enough history for EMA-50 + rolling IV calculations.
        """
        try:
            start = datetime.now(timezone.utc) - timedelta(days=period_days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                feed="iex",
            )
            bars = _with_timeout(self.alpaca_data.get_stock_bars, req)
            df = bars.df
            if df is not None and not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level="symbol")
                df.index = pd.to_datetime(df.index)
                df.columns = [c.lower() for c in df.columns]
                logger.debug(f"{symbol}: {len(df)} bars from Alpaca")
                return df
        except Exception as e:
            logger.warning(f"{symbol}: Alpaca bars failed ({e}), falling back to yfinance")

        return self._yfinance_bars(symbol, period_days)

    def _yfinance_bars(self, symbol: str, period_days: int) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{period_days}d", interval="1d")
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            logger.debug(f"{symbol}: {len(df)} bars from yfinance")
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"{symbol}: yfinance bars failed: {e}")
            return None

    def get_hourly_bars(self, symbol: str, days: int = 15) -> Optional[pd.DataFrame]:
        """
        Fetch 1-hour OHLCV bars for intraday MACD timing signals.
        15 days × 6.5 hours = ~100 bars, enough for MACD (needs 35+).
        """
        try:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                feed="iex",
            )
            bars = _with_timeout(self.alpaca_data.get_stock_bars, req)
            df = bars.df
            if df is not None and not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level="symbol")
                df.index = pd.to_datetime(df.index)
                df.columns = [c.lower() for c in df.columns]
                logger.debug(f"{symbol}: {len(df)} hourly bars from Alpaca")
                return df
        except Exception as e:
            logger.debug(f"{symbol}: Alpaca hourly bars failed ({e}), trying yfinance")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval="60m")
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                logger.debug(f"{symbol}: {len(df)} hourly bars from yfinance")
                return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"{symbol}: Hourly bars failed entirely: {e}")
        return None

    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get the latest bid/ask/mid price quote from Alpaca."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = _with_timeout(self.alpaca_data.get_stock_latest_quote, req)
            q = quote[symbol]
            mid = (q.ask_price + q.bid_price) / 2 if q.ask_price and q.bid_price else None
            return {
                "ask_price": q.ask_price,
                "bid_price": q.bid_price,
                "mid_price": mid,
            }
        except Exception as e:
            logger.warning(f"{symbol}: Quote fetch failed: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for position monitoring."""
        quote = self.get_latest_quote(symbol)
        if quote and quote.get("mid_price"):
            return quote["mid_price"]
        # Fall back to yfinance last close
        try:
            hist = yf.Ticker(symbol).history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ── News Data ────────────────────────────────────────────────────

    def get_company_news(self, symbol: str, hours_back: int = 8) -> List[Dict]:
        """
        Fetch recent news via Alpaca News API (/v1beta1/news).
        Same credentials as the trading client — no extra cost or key needed.
        """
        try:
            start_dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            resp = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers=self._alpaca_news_headers,
                params={
                    "symbols": symbol,
                    "limit": 10,
                    "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "sort": "desc",
                },
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning(f"{symbol}: Alpaca news returned HTTP {resp.status_code}")
                return []

            articles = resp.json().get("news", [])
            filtered = []
            for a in articles:
                if not a.get("headline"):
                    continue
                created = a.get("created_at", "")
                try:
                    ts = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
                except Exception:
                    ts = 0
                filtered.append({
                    "id": a.get("id"),
                    "headline": a.get("headline", ""),
                    "source": a.get("source", ""),
                    "url": a.get("url", ""),
                    "summary": a.get("summary", ""),
                    "datetime": ts,
                    "symbol": symbol,
                })

            logger.debug(f"{symbol}: {len(filtered)} news items in last {hours_back}h (Alpaca)")
            return filtered

        except Exception as e:
            logger.warning(f"{symbol}: Alpaca news fetch failed: {e}")
            return []

    def get_bulk_historical(self, symbols: List[str], period_days: int = 65) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols. Returns {symbol: DataFrame}."""
        result = {}
        for symbol in symbols:
            df = self.get_historical_bars(symbol, period_days)
            if df is not None and len(df) >= 55:
                result[symbol] = df
            time.sleep(0.1)
        logger.info(f"Loaded data for {len(result)}/{len(symbols)} symbols")
        return result
