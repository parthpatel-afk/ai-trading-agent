"""
AI Investor v2 — Data Pipeline
Daily price bars + latest quotes. Alpaca primary, yfinance fallback.

Trimmed for the EMA strategy: no news, no intraday/hourly bars — the bot only
needs daily OHLCV (for EMAs) and a current price/quote (for sizing + monitoring).
"""
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger

from agent.config import config

_ALPACA_DATA_TIMEOUT = 5  # seconds before falling back to yfinance


def _with_timeout(fn, *args, timeout=_ALPACA_DATA_TIMEOUT, **kwargs):
    """Run fn with a wall-clock timeout. Raises TimeoutError on hang."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result(timeout=timeout)


class DataPipeline:
    """Fetches and normalizes daily market data. Returns clean DataFrames."""

    def __init__(self):
        self.alpaca_data = StockHistoricalDataClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )
        logger.info("DataPipeline initialized")

    # ── Historical Daily Bars ─────────────────────────────────────────

    def get_daily_bars(self, symbol: str, period_days: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV bars via Alpaca, with yfinance fallback.
        Default 200 days gives a well-warmed EMA-21 plus the lookback window.
        Returns a DataFrame indexed by date (oldest→newest) with lowercase columns.
        """
        try:
            start = datetime.now(timezone.utc) - timedelta(days=period_days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
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
                logger.debug(f"{symbol}: {len(df)} daily bars from Alpaca")
                return df
        except Exception as e:
            logger.warning(f"{symbol}: Alpaca bars failed ({e}), falling back to yfinance")

        return self._yfinance_bars(symbol, period_days)

    def _yfinance_bars(self, symbol: str, period_days: int) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{period_days}d", interval="1d")
            if df is None or df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            logger.debug(f"{symbol}: {len(df)} daily bars from yfinance")
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"{symbol}: yfinance bars failed: {e}")
            return None

    # ── Quotes / Current Price ────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Latest bid/ask/mid from Alpaca."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = _with_timeout(self.alpaca_data.get_stock_latest_quote, req)
            q = quote[symbol]
            mid = (q.ask_price + q.bid_price) / 2 if q.ask_price and q.bid_price else None
            return {"ask_price": q.ask_price, "bid_price": q.bid_price, "mid_price": mid}
        except Exception as e:
            logger.warning(f"{symbol}: Quote fetch failed: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Most recent price (mid quote, else yfinance last close)."""
        quote = self.get_latest_quote(symbol)
        if quote and quote.get("mid_price"):
            return quote["mid_price"]
        try:
            hist = yf.Ticker(symbol).history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None
