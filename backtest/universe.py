"""
The selectable universe of liquid, optionable US underlyings for the backtester / product.

~50 of the most actively-optioned US names + major ETFs, so a user of the eventual product
can pick which ones to watch. The 18 that mirror the live bot's watchlist are tagged
`default_watchlist=True` (these are what ships as the default strategy).

`sector` drives the direction/sector correlation caps (mirrors agent.config sectors and
extends them). Selection here is by options liquidity + sector spread, NOT a view on the names.
"""
from dataclasses import dataclass
from typing import List, Dict

# Symbols in the live bot's watchlist today (agent/config.py) — kept as the shipped default.
_DEFAULT = {
    "META", "PLTR", "HOOD", "MSFT", "SPY", "IWM", "QQQ", "NVDA", "TSLA",
    "AAPL", "GOOG", "AMZN", "AMD", "TSM", "UNH", "IONQ", "WMT", "JPM",
}


@dataclass(frozen=True)
class Ticker:
    symbol: str
    name: str
    sector: str
    asset: str = "stock"          # "stock" | "etf"
    default_watchlist: bool = False


# (symbol, name, sector, asset). default_watchlist is filled from _DEFAULT below.
_ROWS = [
    # ── Indexes / broad + sector ETFs (deepest option books) ──
    ("SPY",  "S&P 500 ETF",              "index",          "etf"),
    ("QQQ",  "Nasdaq-100 ETF",           "index",          "etf"),
    ("IWM",  "Russell 2000 ETF",         "index",          "etf"),
    ("SMH",  "Semiconductor ETF",        "semis",          "etf"),
    ("XLE",  "Energy Sector ETF",        "energy",         "etf"),
    ("GLD",  "Gold ETF",                 "commodity",      "etf"),
    ("TLT",  "20+yr Treasury ETF",       "rates",          "etf"),
    ("ARKK", "ARK Innovation ETF",       "growth_tech",    "etf"),
    # ── Mega-cap tech / software ──
    ("AAPL", "Apple",                    "megacap_tech"),
    ("MSFT", "Microsoft",                "megacap_tech"),
    ("GOOG", "Alphabet",                 "megacap_tech"),
    ("AMZN", "Amazon",                   "megacap_tech"),
    ("META", "Meta Platforms",           "megacap_tech"),
    ("NFLX", "Netflix",                  "megacap_tech"),
    ("CRM",  "Salesforce",               "megacap_tech"),
    ("ORCL", "Oracle",                   "megacap_tech"),
    ("ADBE", "Adobe",                    "megacap_tech"),
    # ── Semiconductors ──
    ("NVDA", "NVIDIA",                   "semis"),
    ("AMD",  "Advanced Micro Devices",   "semis"),
    ("TSM",  "Taiwan Semiconductor",     "semis"),
    ("AVGO", "Broadcom",                 "semis"),
    ("MU",   "Micron",                   "semis"),
    ("QCOM", "Qualcomm",                 "semis"),
    ("ARM",  "Arm Holdings",             "semis"),
    ("INTC", "Intel",                    "semis"),
    # ── Growth / fintech / high-beta ──
    ("PLTR", "Palantir",                 "growth_tech"),
    ("IONQ", "IonQ",                     "growth_tech"),
    ("TSLA", "Tesla",                    "growth_tech"),
    ("UBER", "Uber",                     "growth_tech"),
    ("ABNB", "Airbnb",                   "growth_tech"),
    ("SHOP", "Shopify",                  "growth_tech"),
    ("SNOW", "Snowflake",                "growth_tech"),
    ("HOOD", "Robinhood",                "fintech"),
    ("COIN", "Coinbase",                 "fintech"),
    ("PYPL", "PayPal",                   "fintech"),
    # ── Financials / payments ──
    ("JPM",  "JPMorgan Chase",           "financials"),
    ("BAC",  "Bank of America",          "financials"),
    ("GS",   "Goldman Sachs",            "financials"),
    ("V",    "Visa",                     "payments"),
    ("MA",   "Mastercard",               "payments"),
    # ── Healthcare ──
    ("UNH",  "UnitedHealth",             "healthcare"),
    ("LLY",  "Eli Lilly",                "healthcare"),
    ("PFE",  "Pfizer",                   "healthcare"),
    ("JNJ",  "Johnson & Johnson",        "healthcare"),
    # ── Energy ──
    ("XOM",  "Exxon Mobil",              "energy"),
    ("CVX",  "Chevron",                  "energy"),
    # ── Industrials ──
    ("BA",   "Boeing",                   "industrials"),
    ("CAT",  "Caterpillar",              "industrials"),
    # ── Consumer / staples ──
    ("WMT",  "Walmart",                  "staples"),
    ("COST", "Costco",                   "staples"),
    ("DIS",  "Disney",                   "consumer"),
    # ── China / EV ──
    ("BABA", "Alibaba",                  "china_tech"),
    ("NIO",  "NIO",                      "china_ev"),
]

UNIVERSE: List[Ticker] = [
    Ticker(sym, name, sector, (rest[0] if rest else "stock"), sym in _DEFAULT)
    for sym, name, sector, *rest in _ROWS
]

_BY_SYMBOL: Dict[str, Ticker] = {t.symbol: t for t in UNIVERSE}


def symbols() -> List[str]:
    """All ~50 selectable symbols."""
    return [t.symbol for t in UNIVERSE]


def default_watchlist() -> List[str]:
    """The 18 that mirror the live bot (the shipped default strategy)."""
    return [t.symbol for t in UNIVERSE if t.default_watchlist]


def sector_of(symbol: str) -> str:
    t = _BY_SYMBOL.get(symbol)
    return t.sector if t else "other"


def get(symbol: str) -> Ticker:
    return _BY_SYMBOL[symbol]


if __name__ == "__main__":
    print(f"Universe: {len(UNIVERSE)} names ({len(default_watchlist())} default watchlist)")
    by_sector: Dict[str, List[str]] = {}
    for t in UNIVERSE:
        by_sector.setdefault(t.sector, []).append(t.symbol)
    for sec in sorted(by_sector):
        print(f"  {sec:14} {' '.join(by_sector[sec])}")
