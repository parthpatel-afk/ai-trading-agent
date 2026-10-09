"""
AI Investor Agent - Groq Pre-Screener
Layer 1 of the multi-model pipeline. Runs BEFORE any expensive API calls.
Uses Groq's free Llama 3.3 70B (sub-second latency) to ask:
"Is this stock worth a full analysis right now?"
If NO → skip the symbol entirely, saving Gemini + GPT-4o-mini + GPT-4o calls.
"""
import json
from dataclasses import dataclass
from groq import Groq
from loguru import logger

from agent.config import config
from agent.signals.technical import TechnicalSignals


@dataclass
class PreScreenResult:
    symbol: str
    worth_analyzing: bool
    reason: str
    confidence: float = 0.0     # 0–1, how confident Groq is in this call
    skipped_by: str = "groq"


def pre_screen(symbol: str, technical: TechnicalSignals) -> PreScreenResult:
    """
    Quick Groq/Llama check: is this symbol worth running the full pipeline on?
    Takes ~0.5 seconds. Saves ~3-4 API calls per skipped symbol.

    Returns PreScreenResult with worth_analyzing=True/False.
    If Groq is unavailable (no key, rate limit), defaults to True (don't block analysis).
    """
    if not config.groq.api_key:
        return PreScreenResult(symbol=symbol, worth_analyzing=True,
                               reason="Groq not configured — passing through", confidence=0.5)

    prompt = f"""You are a fast options trade pre-screener. Analyze these technical signals for {symbol}
and decide if it's worth a full AI analysis (sentiment + GPT-4o brain).

Technical snapshot:
- RSI: {technical.rsi} (valid range: 30-72, extremes above 78 or below 25 are bad)
- MACD bullish (above signal line): {technical.macd_bullish}
- MACD recent crossover (last 7 days): {technical.macd_crossover}
- Price above 50-EMA (uptrend confirmed): {technical.above_ema_50}
- Volume vs 20-day avg: {technical.volume_ratio}x (normal volume >= 1.0x is fine)
- Bollinger %B: {technical.bollinger_pct} (0=at lower band, 1=at upper band, 0.2-0.7 is healthy)
- Technical score: {technical.technical_score:.2f}/1.0 (above 0.45 is worth analyzing)
- Entry signal triggered: {technical.entry_signal}

Respond ONLY with valid JSON:
{{"worth_analyzing": true/false, "reason": "one sentence max", "confidence": 0.0-1.0}}

Say false ONLY if: RSI is extreme (>78 or <25) OR the stock is in a clear downtrend (below 50-EMA AND MACD falling AND volume declining simultaneously). Do NOT reject based on technical score — that was already checked upstream.
Say true in all other cases. This stock has already passed a strict technical filter."""

    try:
        client = Groq(api_key=config.groq.api_key)
        response = client.chat.completions.create(
            model=config.groq.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=80,
            temperature=0.1,
        )
        result = json.loads(response.choices[0].message.content)
        worth = bool(result.get("worth_analyzing", True))
        reason = result.get("reason", "")
        confidence = float(result.get("confidence", 0.5))

        if not worth:
            logger.debug(f"{symbol}: Pre-screener SKIP — {reason} (conf={confidence:.2f})")
        else:
            logger.debug(f"{symbol}: Pre-screener PASS — {reason}")

        return PreScreenResult(
            symbol=symbol,
            worth_analyzing=worth,
            reason=reason,
            confidence=confidence,
        )

    except Exception as e:
        logger.debug(f"{symbol}: Groq pre-screener error ({e}) — defaulting to pass")
        return PreScreenResult(symbol=symbol, worth_analyzing=True,
                               reason=f"Groq unavailable: {str(e)[:60]}", confidence=0.5)
