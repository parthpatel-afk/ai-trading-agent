"""
AI Investor Agent - Gemini Flash Sanity Check
Layer 3 of the multi-model pipeline. Runs AFTER signals, BEFORE GPT-4o brain decision.
Uses Google Gemini Flash (free tier) to catch anything the numbers miss:
- Macro news the sentiment model didn't pick up
- Sector-wide risk (e.g., regulatory news hitting a whole industry)
- Pattern conflicts the composite score can't see

All candidates are sent in ONE Gemini call to avoid 503s from rapid sequential requests.
If Gemini is unavailable, all candidates pass (fail-open) — brain still works without it.
"""
import json
from dataclasses import dataclass
from typing import Dict, List
from google import genai
from google.genai import types
from loguru import logger

from agent.config import config


@dataclass
class SanityCheckResult:
    symbol: str
    context: str        # Plain-text paragraph passed to GPT-4o brain
    red_flags: list     # List of specific concerns (may be empty)
    passed: bool        # True = no major red flags found
    from_gemini: bool = True


def _make_fallback(symbol: str) -> SanityCheckResult:
    return SanityCheckResult(symbol=symbol, context="", red_flags=[], passed=True, from_gemini=False)


def run_batch_sanity_check(bundles: List[dict]) -> Dict[str, SanityCheckResult]:
    """
    Send ALL pipeline candidates to Gemini in one call.
    Returns a dict of symbol -> SanityCheckResult.
    If Gemini is unavailable, all candidates pass (fail-open).
    """
    if not config.gemini.api_key or not bundles:
        return {b["symbol"]: _make_fallback(b["symbol"]) for b in bundles}

    # Build per-candidate signal blocks
    candidate_blocks = []
    for b in bundles:
        tech = b["technical"]
        sent = b["sentiment"]
        macro = b["macro"]
        opts = b["options_flow"]
        w = config.strategy
        w_total = w.weight_technical + w.weight_sentiment + w.weight_macro + w.weight_options_flow
        composite = (
            tech.technical_score * (w.weight_technical / w_total) +
            ((sent.overall_score + 1.0) / 2.0) * (w.weight_sentiment / w_total) +
            macro.regime_score * (w.weight_macro / w_total) +
            opts.options_score * (w.weight_options_flow / w_total)
        )
        candidate_blocks.append(
            f"[{b['symbol']}]\n"
            f"Composite: {composite:.2f} | Technical: {tech.technical_score:.2f} | "
            f"RSI: {tech.rsi:.0f} | MACD crossover: {tech.macd_crossover} | Above 50-EMA: {tech.above_ema_50}\n"
            f"Sentiment: {sent.overall_score:.2f} ({sent.signal_strength}) | {sent.news_count} articles | {sent.summary}\n"
            f"Macro: {macro.regime} | VIX: {macro.vix_level}\n"
            f"IV Rank: {opts.iv_rank} ({opts.iv_regime}) | P/C ratio: {opts.put_call_ratio} | "
            f"Strategy: {opts.strategy_mode}"
        )

    prompt = (
        "You are a senior risk analyst reviewing multiple proposed options trades. "
        "For each stock, flag anything suspicious, contradictory, or risky. "
        "Respond with ONLY valid JSON — no markdown, no explanation outside the JSON.\n\n"
        "Format: {\"results\": [{\"symbol\": \"X\", \"risk_level\": \"LOW|MEDIUM|HIGH\", "
        "\"red_flags\": [], \"context\": \"2-3 sentence assessment\"}]}\n\n"
        "Candidates:\n\n" + "\n\n".join(candidate_blocks)
    )

    try:
        client = genai.Client(api_key=config.gemini.api_key)
        response = client.models.generate_content(
            model=config.gemini.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=600,
                temperature=0.2,
            )
        )
        raw = response.text.strip() if response.text else ""

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        results = {}
        for item in data.get("results", []):
            sym = item.get("symbol", "")
            risk_level = item.get("risk_level", "LOW").upper()
            red_flags = item.get("red_flags", [])
            context = item.get("context", "")
            passed = risk_level != "HIGH"
            results[sym] = SanityCheckResult(
                symbol=sym,
                context=context,
                red_flags=red_flags,
                passed=passed,
                from_gemini=True,
            )
            logger.debug(f"{sym}: Gemini sanity | risk={risk_level} | flags={red_flags}")

        # Fill in any symbols Gemini missed (fail-open)
        for b in bundles:
            if b["symbol"] not in results:
                logger.debug(f"{b['symbol']}: Gemini response missing — fail open")
                results[b["symbol"]] = _make_fallback(b["symbol"])

        logger.info(f"Gemini batch sanity: {len(bundles)} candidates, 1 call")
        return results

    except Exception as e:
        logger.warning(f"Gemini batch sanity check failed ({e}) — all candidates pass")
        return {b["symbol"]: _make_fallback(b["symbol"]) for b in bundles}
