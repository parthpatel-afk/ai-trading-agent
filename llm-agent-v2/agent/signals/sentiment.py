"""
AI Investor Agent - Sentiment Signal Module v2.0
Cost-efficient: ONE batched GPT-4o-mini call per stock (vs 11 calls before).
Results cached 2 hours so repeated scans don't re-analyze the same news.
~96% cost reduction vs v1.0 with comparable accuracy.
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
from loguru import logger

from agent.config import config

# ── In-memory sentiment cache: {symbol -> (SentimentSignals, timestamp, fingerprint)} ──
# fingerprint = the set of article IDs/headlines that produced the score. We reuse a
# cached score only while the underlying news is UNCHANGED (and within the TTL backstop).
# The instant a new article appears or an old one ages out of the lookback window, the
# fingerprint changes and we re-score — so an actively-watched name is never gated by a
# stale read (the 2026-06-03 AAPL case: a 2h-old −0.25 blocked entry until the cache
# happened to expire, by which point the name had already run).
_cache: Dict[str, tuple] = {}


def _news_fingerprint(news_items: List[Dict]) -> tuple:
    """Stable fingerprint of a news set — by article id, falling back to headline text."""
    keys = []
    for item in news_items[:8]:
        k = item.get("id") or item.get("headline", "")
        if k:
            keys.append(str(k))
    return tuple(sorted(keys))


@dataclass
class SentimentSignals:
    symbol: str
    overall_score: float = 0.0          # Weighted avg: -1.0 to +1.0
    news_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    summary: str = ""
    signal_strength: str = "none"       # "strong_positive"|"positive"|"neutral"|"negative"|"strong_negative"
    hours_back: int = 8
    from_cache: bool = False
    scored_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def score_sentiment(symbol: str, news_items: List[Dict]) -> SentimentSignals:
    """
    Score news sentiment for a stock.
    Single batched GPT-4o-mini call covers all headlines.
    Results are cached for sentiment_cache_minutes to avoid redundant API calls.
    """
    # ── Cache check — reuse ONLY if the news set is unchanged (and within TTL) ──
    cache_ttl_seconds = config.strategy.sentiment_cache_minutes * 60
    fingerprint = _news_fingerprint(news_items)
    if symbol in _cache:
        cached_result, cached_at, cached_fp = _cache[symbol]
        fresh_enough = time.time() - cached_at < cache_ttl_seconds
        if fresh_enough and cached_fp == fingerprint:
            logger.debug(
                f"{symbol}: Sentiment from cache (news unchanged, "
                f"age={(time.time()-cached_at)/60:.1f}min)"
            )
            cached_result.from_cache = True
            return cached_result
        if not fresh_enough:
            logger.debug(f"{symbol}: Sentiment cache expired (TTL) — re-scoring")
        elif cached_fp != fingerprint:
            logger.debug(f"{symbol}: News changed since last score — re-scoring fresh")

    if not news_items:
        result = SentimentSignals(
            symbol=symbol,
            overall_score=0.0,
            signal_strength="none",
            summary="No recent news found."
        )
        _cache[symbol] = (result, time.time(), fingerprint)
        return result

    client = OpenAI(api_key=config.openai.api_key)

    HIGH_CREDIBILITY_SOURCES = [
        "reuters", "bloomberg", "wsj", "ft", "cnbc", "marketwatch",
        "barrons", "yahoo finance", "ap", "associated press", "seekingalpha"
    ]

    # Cap at 8 most recent articles (enough signal, keeps prompt tight)
    articles = news_items[:8]

    # ── Build batched prompt ──────────────────────────────────────────
    headlines_block = "\n".join(
        f"{i+1}. [{item.get('source', 'unknown')}] {item.get('headline', '')}"
        for i, item in enumerate(articles)
        if item.get("headline")
    )

    if not headlines_block.strip():
        result = SentimentSignals(symbol=symbol, overall_score=0.0, signal_strength="none",
                                  summary="No valid headlines found.")
        _cache[symbol] = (result, time.time(), fingerprint)
        return result

    system_prompt = (
        f"You are a financial sentiment analyst. Score the following {len(articles)} news headlines "
        f"for stock ticker {symbol}. "
        "Return ONLY valid JSON with this exact structure:\n"
        '{"scores": [<float -1 to 1 per headline>], '
        '"summary": "<1-2 sentence overall sentiment summary>", '
        '"overall": <float -1 to 1>}'
        "\nScore guide: -1.0=very bearish, -0.5=bearish, 0=neutral, 0.5=bullish, 1.0=very bullish."
    )

    user_prompt = f"Headlines for {symbol}:\n{headlines_block}"

    try:
        response = client.chat.completions.create(
            model=config.openai.sentiment_model,  # gpt-4o-mini
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1,
        )
        result_json = json.loads(response.choices[0].message.content)
        individual_scores = result_json.get("scores", [])
        summary = result_json.get("summary", "")
        overall_raw = result_json.get("overall", 0.0)

    except Exception as e:
        logger.warning(f"{symbol}: Sentiment API call failed: {e}")
        individual_scores = []
        summary = f"Sentiment unavailable: {str(e)[:80]}"
        overall_raw = 0.0

    # ── Apply source credibility weights ─────────────────────────────
    weighted_scores = []
    total_weight = 0.0
    for i, item in enumerate(articles):
        source = item.get("source", "").lower()
        weight = 1.5 if any(src in source for src in HIGH_CREDIBILITY_SOURCES) else 1.0
        raw_score = individual_scores[i] if i < len(individual_scores) else overall_raw
        raw_score = max(-1.0, min(1.0, float(raw_score)))
        weighted_scores.append(raw_score * weight)
        total_weight += weight

    if weighted_scores and total_weight > 0:
        overall = sum(weighted_scores) / total_weight
    else:
        overall = max(-1.0, min(1.0, float(overall_raw)))
    overall = round(overall, 3)

    # ── Count sentiment breakdown ─────────────────────────────────────
    pos = sum(1 for s in individual_scores if s > 0.15)
    neg = sum(1 for s in individual_scores if s < -0.15)
    neu = len(individual_scores) - pos - neg

    # ── Signal strength label ─────────────────────────────────────────
    if overall >= 0.50:
        strength = "strong_positive"
    elif overall >= 0.20:
        strength = "positive"
    elif overall <= -0.50:
        strength = "strong_negative"
    elif overall <= -0.20:
        strength = "negative"
    else:
        strength = "neutral"

    signal = SentimentSignals(
        symbol=symbol,
        overall_score=overall,
        news_count=len(articles),
        positive_count=pos,
        negative_count=neg,
        neutral_count=neu,
        summary=summary,
        signal_strength=strength,
        hours_back=config.strategy.news_lookback_hours,
        from_cache=False,
    )

    # ── Store in cache ────────────────────────────────────────────────
    _cache[symbol] = (signal, time.time(), fingerprint)
    logger.info(f"{symbol}: Sentiment={overall:.3f} ({strength}), {len(articles)} articles [1 API call]")
    return signal


def clear_sentiment_cache(symbol: Optional[str] = None):
    """Clear cache for one symbol or all."""
    if symbol:
        _cache.pop(symbol, None)
    else:
        _cache.clear()
