"""
AI Investor Agent - Brain v2.0 (Options Decision Layer)
Synthesizes all 4 signals into a structured options trading decision via GPT-4o.
Dual-mode: directional (buy calls/puts) or defined-risk (spreads).
Signal weights are normalized so composite always maxes at 1.0.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from loguru import logger

from agent.config import config
from agent.signals.technical import TechnicalSignals
from agent.signals.sentiment import SentimentSignals
from agent.signals.macro import MacroSignals
from agent.signals.options_flow import OptionsFlowSignals


# Actions that OPEN a new position (directional buys + defined-risk credit sells).
# Used to gate execution and to sort entries ahead of HOLD/SKIP in batch ranking.
ENTRY_ACTIONS = {
    "BUY_CALL", "BUY_PUT",
    "SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "IRON_CONDOR",
}


@dataclass
class BrainDecision:
    """The agent's final options trading decision for one symbol."""
    symbol: str
    action: str                         # "BUY_CALL"|"BUY_PUT"|"HOLD"|"SKIP"
    confidence: float                   # 0.0–1.0
    reasoning: str                      # Full LLM explanation (plain English)

    # Scored signal breakdown (all 0–1 scale)
    technical_score: float = 0.0
    sentiment_score_normalized: float = 0.0  # -1..+1 → 0..1
    macro_score: float = 0.0
    options_flow_score: float = 0.0
    composite_score: float = 0.0

    score_breakdown: Dict[str, Any] = field(default_factory=dict)

    # Options-specific execution parameters (executor uses these)
    strategy_mode: str = "SKIP"        # Mirrors action
    target_dte: int = 35               # Days to expiration to target
    position_size_pct: float = 0.0     # Fraction of account to risk

    # Stop/target as % of premium
    stop_loss_pct: float = 0.50        # Exit if premium drops 50%
    profit_target_pct: float = 1.00    # Exit if premium doubles (100% gain)

    # Sentiment decay fields — set by batch brain from LLM response
    catalyst_type: str = "momentum"         # "fundamental"|"momentum"|"sentiment_buzz"
    sentiment_shelf_life_days: int = 7       # days before re-checking sentiment thesis
    sentiment_score_at_entry: float = 0.0   # raw -1..+1 score at decision time

    # Was this blocked before hitting the LLM?
    blocked: bool = False
    blocked_reason: str = ""


BATCH_SYSTEM_PROMPT = """You are an expert options trading analyst AI. Multiple stock candidates have
passed all technical, sentiment, options, and sanity filters. Each candidate already has a
specific option contract identified — you are deciding whether to BUY that exact contract.

Your decisions must be:
1. Comparative — evaluate candidates head-to-head, not each in isolation
2. Contract-aware — the breakeven % tells you exactly how much the stock must move to profit;
   high breakeven moves require stronger conviction; prefer contracts where the stock can
   realistically reach breakeven within the DTE window
3. Data-driven — reference specific signal values AND contract mechanics in reasoning
4. Options-aware — prefer lower IV rank (cheaper to buy premium) and stronger momentum
5. Concise — 2-3 sentences per symbol maximum

Each candidate arrives pre-tagged with a recommended `mode`:
- BUY_CALL / BUY_PUT — a directional debit (we PAY premium). Best when IV is cheap.
- SELL_PUT_SPREAD / SELL_CALL_SPREAD — a defined-risk CREDIT spread (we COLLECT premium and
  profit from time decay; max loss is capped). Recommended when IV is rich. For these, the
  candidate block shows the credit, the width, and credit/width — higher credit/width and a
  lower short-leg delta (further OTM) is a higher-probability sell. Do NOT reason about
  "breakeven move" for these the way you would for a long option — they win if the stock simply
  does NOT blow past the short strike.
- IRON_CONDOR — sell both a put spread and a call spread (range-bound).

Keep the candidate's recommended `mode` as the action unless the signals clearly contradict it;
your job is to rank conviction and pick the best, not to override the strategy type.

You MUST respond with ONLY valid JSON in this exact format:
{
  "decisions": [
    {
      "symbol": "<ticker>",
      "action": "BUY_CALL" | "BUY_PUT" | "SELL_PUT_SPREAD" | "SELL_CALL_SPREAD" | "IRON_CONDOR" | "SKIP",
      "confidence": <float 0.0-1.0>,
      "reasoning": "<2-3 sentences referencing the mechanics that fit this strategy type>",
      "target_dte": <int 21-45>,
      "position_size_pct": <float 0.03-0.08>,
      "catalyst_type": "fundamental" | "momentum" | "sentiment_buzz",
      "sentiment_shelf_life_days": <int 3-15>
    }
  ]
}

catalyst_type guide (shelf life drives when to re-check sentiment and exit if buzz faded):
- "fundamental": earnings beat, analyst upgrade, guidance raise, macro tailwind → shelf_life 10-15
- "momentum": technical breakout, sector rotation, high-volume surge → shelf_life 5-10
- "sentiment_buzz": product launch, trending headline, short-term news event → shelf_life 3-5

Include ALL candidates in the decisions array. Selected candidates keep their recommended entry
mode; unselected candidates get SKIP with a brief reason. Confidence >= 0.60 required for any entry.
"""

SYSTEM_PROMPT = """You are an expert options trading analyst AI. You analyze stocks using technical
indicators, news sentiment, market regime data, and options-specific signals (IV Rank, earnings)
to make clear options trading decisions.

Your decisions must be:
1. Data-driven — reference specific indicator values in your reasoning
2. Concise — 3-4 sentences maximum
3. Options-aware — consider IV rank (cheap = better to BUY premium)
4. Risk-aware — never recommend oversized positions, respect earnings blackout

You MUST respond with ONLY valid JSON matching this exact schema:
{
  "action": "BUY_CALL" | "BUY_PUT" | "HOLD" | "SKIP",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<plain English, 3-4 sentences>",
  "target_dte": <int, recommended days to expiration, between 21-45>,
  "position_size_pct": <float 0.03-0.08, fraction of account to risk>,
  "stop_loss_pct": <float, typically 0.50 for long options>,
  "profit_target_pct": <float, typically 1.00 for long options>
}

Action guide:
- BUY_CALL: Bullish signal, confidence >= 0.60
- BUY_PUT: Bearish signal, confidence >= 0.60
- HOLD: Open position still valid, no exit triggered yet
- SKIP: Mixed signals, earnings risk, low confidence, or extreme macro conditions
"""


class Brain:
    def __init__(self):
        self.client = OpenAI(api_key=config.openai.api_key)
        self.cfg = config.strategy
        self.opts = config.options
        logger.info("Brain v2.0 (options mode) initialized")

    def decide(
        self,
        symbol: str,
        technical: TechnicalSignals,
        sentiment: SentimentSignals,
        macro: MacroSignals,
        options_flow: OptionsFlowSignals,
        current_price: float,
        has_open_position: bool = False,
    ) -> BrainDecision:
        """
        Core decision function. Synthesizes all 4 signals → LLM reasoning → decision.
        Returns a fully scored, logged BrainDecision.
        """

        # ── Step 1: Compute normalized scores ────────────────────────
        tech_score = technical.technical_score                      # Already 0–1
        sent_score_norm = (sentiment.overall_score + 1.0) / 2.0    # -1..+1 → 0..1
        macro_score = macro.regime_score                            # Already 0–1
        opts_score = options_flow.options_score                     # Already 0–1

        # ── Step 2: Composite score — weights sum to 1.0 ─────────────
        # Normalization guard: if weights don't sum to 1.0 exactly, normalize
        w_tech = self.cfg.weight_technical
        w_sent = self.cfg.weight_sentiment
        w_macro = self.cfg.weight_macro
        w_opts = self.cfg.weight_options_flow
        w_total = w_tech + w_sent + w_macro + w_opts

        composite = (
            tech_score * (w_tech / w_total) +
            sent_score_norm * (w_sent / w_total) +
            macro_score * (w_macro / w_total) +
            opts_score * (w_opts / w_total)
        )
        composite = round(composite, 3)

        score_breakdown = {
            "technical": {
                "score": round(tech_score, 3),
                "weight": w_tech,
                "contribution": round(tech_score * (w_tech / w_total), 3),
                "rsi": technical.rsi,
                "macd_crossover": technical.macd_crossover,
                "above_ema_50": technical.above_ema_50,
                "volume_ratio": technical.volume_ratio,
            },
            "sentiment": {
                "score": round(sent_score_norm, 3),
                "raw_score": sentiment.overall_score,
                "weight": w_sent,
                "contribution": round(sent_score_norm * (w_sent / w_total), 3),
                "strength": sentiment.signal_strength,
                "news_count": sentiment.news_count,
                "from_cache": sentiment.from_cache,
                "summary": sentiment.summary,
            },
            "macro": {
                "score": round(macro_score, 3),
                "weight": w_macro,
                "contribution": round(macro_score * (w_macro / w_total), 3),
                "regime": macro.regime,
                "vix": macro.vix_level,
                "notes": macro.notes,
            },
            "options_flow": {
                "score": round(opts_score, 3),
                "weight": w_opts,
                "contribution": round(opts_score * (w_opts / w_total), 3),
                "iv_rank": options_flow.iv_rank,
                "iv_regime": options_flow.iv_regime,
                "put_call_ratio": options_flow.put_call_ratio,
                "earnings_blackout": options_flow.earnings_within_blackout,
                "strategy_mode": options_flow.strategy_mode,
                "notes": options_flow.notes,
            },
            "composite": composite,
        }

        # ── Step 3: Hard pre-checks (block before calling LLM) ───────
        if options_flow.earnings_within_blackout and not has_open_position:
            return BrainDecision(
                symbol=symbol, action="SKIP", confidence=0.0,
                reasoning=f"Earnings in {options_flow.earnings_date} — earnings blackout active. Binary risk too high for options.",
                composite_score=composite, score_breakdown=score_breakdown,
                blocked=True, blocked_reason="earnings_blackout",
                technical_score=tech_score, sentiment_score_normalized=sent_score_norm,
                macro_score=macro_score, options_flow_score=opts_score,
            )

        if not macro.allow_new_entries and not has_open_position:
            return BrainDecision(
                symbol=symbol, action="SKIP", confidence=0.0,
                reasoning=f"Market in {macro.regime} regime — new option entries paused. Protecting capital.",
                composite_score=composite, score_breakdown=score_breakdown,
                blocked=True, blocked_reason="macro_circuit_breaker",
                technical_score=tech_score, sentiment_score_normalized=sent_score_norm,
                macro_score=macro_score, options_flow_score=opts_score,
            )

        if composite < self.cfg.min_signal_confidence and not has_open_position:
            return BrainDecision(
                symbol=symbol, action="SKIP", confidence=composite,
                reasoning=f"Composite score {composite:.2f} below minimum {self.cfg.min_signal_confidence}. Signals too weak to risk premium.",
                composite_score=composite, score_breakdown=score_breakdown,
                blocked=True, blocked_reason="low_confidence",
                technical_score=tech_score, sentiment_score_normalized=sent_score_norm,
                macro_score=macro_score, options_flow_score=opts_score,
            )

        # ── Step 4: Build LLM context prompt ─────────────────────────
        user_prompt = f"""
Analyze {symbol} for an options trading decision. Current price: ${current_price:.2f}
Has open position: {has_open_position}
Recommended strategy from pre-analysis: {options_flow.strategy_mode}

TECHNICAL SIGNALS (score: {tech_score:.2f}/1.0):
- RSI: {technical.rsi} ({technical.rsi_signal})
- MACD Crossover (bullish): {technical.macd_crossover}
- MACD Bullish: {technical.macd_bullish}
- Price above 50-EMA: {technical.above_ema_50}
- Volume ratio vs 20d avg: {technical.volume_ratio}x
- Bollinger %B: {technical.bollinger_pct}
- ATR: ${technical.atr} ({technical.atr_pct}% of price)
- Entry signal: {technical.entry_signal}

SENTIMENT (score: {sentiment.overall_score:.2f}, strength: {sentiment.signal_strength}):
- News analyzed: {sentiment.news_count} articles
- Pos/Neg/Neutral: {sentiment.positive_count}/{sentiment.negative_count}/{sentiment.neutral_count}
- Summary: {sentiment.summary}

OPTIONS FLOW (score: {opts_score:.2f}/1.0):
- IV Rank: {options_flow.iv_rank} (0=cheapest, 100=most expensive)
- IV Regime: {options_flow.iv_regime} (cheap/normal = buy options; expensive = use spreads)
- Put/Call Ratio: {options_flow.put_call_ratio} (>1.2 bearish, <0.7 bullish)
- Earnings within {self.opts.earnings_blackout_days}d: {options_flow.earnings_within_blackout}
- Notes: {options_flow.notes}

MACRO REGIME (score: {macro_score:.2f}/1.0):
- Regime: {macro.regime}
- VIX: {macro.vix_level}
- SPY trend: {macro.spy_trend}
- Notes: {macro.notes}

COMPOSITE SCORE: {composite:.3f}/1.0 (minimum required: {self.cfg.min_signal_confidence})

"""

        # ── Step 5: Call LLM ──────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model=config.openai.brain_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=450,
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)

            valid_actions = {"BUY_CALL", "BUY_PUT", "HOLD", "SKIP"}
            action = result.get("action", "SKIP").upper()
            if action not in valid_actions:
                action = "SKIP"

            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            reasoning = result.get("reasoning", "No reasoning provided.")
            target_dte = max(self.opts.min_dte, min(self.opts.max_dte, int(result.get("target_dte", 35))))
            pos_size = max(0.03, min(self.cfg.risk_config_max_pct if hasattr(self.cfg, 'risk_config_max_pct') else 0.08,
                                     float(result.get("position_size_pct", 0.06))))
            stop_pct = float(result.get("stop_loss_pct", self.opts.stop_loss_pct))
            tp_pct = float(result.get("profit_target_pct", self.opts.profit_target_long))

            # Apply macro position size multiplier
            pos_size *= macro.position_size_multiplier
            pos_size = round(min(pos_size, config.risk.max_position_pct), 4)

            logger.info(
                f"Brain → {symbol}: {action} (confidence={confidence:.2f}, composite={composite:.3f}, "
                f"mode={options_flow.strategy_mode})"
            )

            return BrainDecision(
                symbol=symbol,
                action=action,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                technical_score=tech_score,
                sentiment_score_normalized=round(sent_score_norm, 3),
                macro_score=round(macro_score, 3),
                options_flow_score=round(opts_score, 3),
                composite_score=composite,
                score_breakdown=score_breakdown,
                strategy_mode=action,
                target_dte=target_dte,
                position_size_pct=pos_size,
                stop_loss_pct=stop_pct,
                profit_target_pct=tp_pct,
            )

        except Exception as e:
            logger.error(f"Brain LLM call failed for {symbol}: {e}")
            return BrainDecision(
                symbol=symbol, action="SKIP", confidence=0.0,
                reasoning=f"LLM decision unavailable: {str(e)[:100]}",
                composite_score=composite, score_breakdown=score_breakdown,
                blocked=True, blocked_reason="llm_error",
                technical_score=tech_score, sentiment_score_normalized=sent_score_norm,
                macro_score=macro_score, options_flow_score=opts_score,
            )

    def review_losing_position(
        self,
        symbol: str,
        option_type: str,
        loss_pct: float,
        days_held: int,
        dte_remaining: int,
        technical: TechnicalSignals,
        news_headlines: List[str],
        sentiment_score: float,
        sentiment_news_count: int,
    ) -> Tuple[bool, str]:
        """
        Layer-3 gut-check on a losing long option (called once when a position first
        crosses the gutcheck loss trigger). Looks at the live technical picture + fresh
        news and decides whether the ORIGINAL thesis is broken (→ cut now) or this is
        just noise (→ hold and let the structural stops manage it).

        Returns (should_cut, reasoning). Fails SAFE: on any error returns (False, ...)
        so a flaky API call never force-closes a position.
        """
        direction = "upward (we own a CALL — we need the stock to RISE)" if option_type == "call" \
            else "downward (we own a PUT — we need the stock to FALL)"
        headlines_block = "\n".join(f"- {h}" for h in news_headlines[:8]) or "- (no fresh headlines found)"

        user_prompt = f"""We own a losing {option_type.upper()} option on {symbol} and must decide: CUT now or HOLD.

Our original thesis was {direction}.

POSITION:
- Down {loss_pct:.0%} of premium paid
- Held {days_held} day(s), {dte_remaining} days to expiry remaining

LIVE TECHNICAL PICTURE (the underlying stock right now):
- Price: ${technical.close} | 50-EMA: ${technical.ema_50} | Price above 50-EMA: {technical.above_ema_50}
- RSI: {technical.rsi} ({technical.rsi_signal})
- MACD bullish (above signal): {technical.macd_bullish} | MACD hist: {technical.macd_hist}
- ATR: ${technical.atr} ({technical.atr_pct}% of price)
- Daily exit signal flagged: {technical.exit_signal} ({technical.exit_reason})

FRESH NEWS (last several hours), sentiment {sentiment_score:+.2f} from {sentiment_news_count} articles:
{headlines_block}

DECISION RULE:
- CUT if the thesis is genuinely broken: the trend that justified the trade has reversed
  (e.g. a CALL whose stock has fallen below its 50-EMA with momentum rolling over), OR there
  is materially bad news (downgrade, guidance cut, bad earnings, sector shock).
- HOLD if this is ordinary noise / a normal pullback with the trend still intact and no real
  bad news — let the mechanical stops manage it.

Respond as JSON: {{"cut": true/false, "reason": "one concise sentence in plain English"}}"""

        try:
            response = self.client.chat.completions.create(
                model=config.openai.brain_model,
                messages=[
                    {"role": "system", "content": "You are a disciplined options risk manager. "
                     "You decide whether to cut a losing position based on whether its thesis is "
                     "broken, not on the size of the loss. Be decisive and concise."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=120,
                temperature=0.1,
            )
            result = json.loads(response.choices[0].message.content)
            should_cut = bool(result.get("cut", False))
            reason = str(result.get("reason", "no reason given"))[:200]
            logger.info(f"Gut-check → {symbol} ({option_type}, -{loss_pct:.0%}): "
                        f"{'CUT' if should_cut else 'HOLD'} — {reason}")
            return should_cut, reason
        except Exception as e:
            logger.warning(f"Gut-check LLM call failed for {symbol}: {e} — holding (fail-safe)")
            return False, f"gut-check unavailable: {str(e)[:80]}"

    def decide_batch(
        self,
        candidates: List[dict],
        available_slots: int,
    ) -> List[Tuple["BrainDecision", dict]]:
        """
        Single GPT-4o call that ranks all pipeline survivors head-to-head and picks the best.
        Returns list of (BrainDecision, bundle) tuples — BUY actions sorted first by confidence.
        """
        if not candidates:
            return []

        # Build per-candidate summary blocks for the prompt
        blocks = []
        for i, c in enumerate(candidates, 1):
            tech = c["technical"]
            sent = c["sentiment"]
            macro = c["macro"]
            opts = c["options_flow"]
            w_tech = self.cfg.weight_technical
            w_sent = self.cfg.weight_sentiment
            w_macro = self.cfg.weight_macro
            w_opts = self.cfg.weight_options_flow
            w_total = w_tech + w_sent + w_macro + w_opts
            # Use the direction-matched strength score (shorts score on the bearish scorer).
            dir_score = c.get("directional_score") or tech.technical_score
            composite = round(
                dir_score * (w_tech / w_total) +
                ((sent.overall_score + 1.0) / 2.0) * (w_sent / w_total) +
                macro.regime_score * (w_macro / w_total) +
                opts.options_score * (w_opts / w_total), 3
            )

            price = c.get("current_price", 0)
            spread = c.get("spread")
            ct = c.get("contract", {})
            if spread:
                # Defined-risk credit spread — show the economics that matter for a SELLER.
                sl = spread.get("short_leg", {})
                contract_block = (
                    f"  CREDIT SPREAD ({spread.get('strategy_type', '')}): "
                    f"exp={spread.get('expiration', 'N/A')} | DTE={spread.get('dte', 0)}\n"
                    f"  Short {sl.get('strike', 0):g}{spread.get('option_type', '')[:1]} "
                    f"(Δ≈{sl.get('abs_delta', 0):.2f}) / Long {spread.get('long_leg', {}).get('strike', 0):g}\n"
                    f"  Credit: ${spread.get('credit_per_spread', 0):.0f} | Width: ${spread.get('width_per_spread', 0):.0f} "
                    f"| credit/width={spread.get('credit_to_width', 0):.0%} "
                    f"| Max loss: ${spread.get('max_loss_per_spread', 0):.0f} | Stock now: ${price:.2f}"
                )
            elif ct:
                contract_block = (
                    f"  CONTRACT: {ct.get('occ_symbol', 'N/A')} | {ct.get('option_type', '')} "
                    f"| strike=${ct.get('strike', 0):.0f} | exp={ct.get('expiration', 'N/A')} | DTE={ct.get('dte', 0)}\n"
                    f"  Cost: ${ct.get('premium_per_contract', 0):.0f}/contract "
                    f"| Bid-Ask: ${ct.get('bid', 0):.2f}-${ct.get('ask', 0):.2f} "
                    f"| OI={ct.get('open_interest', 0)}\n"
                    f"  Stock now: ${price:.2f} | Move to strike: {ct.get('move_needed_pct', 0):+.1f}% "
                    f"| Breakeven: ${ct.get('breakeven', 0):.2f} ({ct.get('breakeven_move_pct', 0):.1f}% move needed)"
                )
            else:
                contract_block = "  CONTRACT: not available"

            blocks.append(
                f"[{i}] {c['symbol']} (composite={composite:.3f}, mode={opts.strategy_mode})\n"
                f"  Technical: score={tech.technical_score:.2f}, RSI={tech.rsi}, "
                f"MACD_bullish={tech.macd_bullish}, above_EMA50={tech.above_ema_50}, vol={tech.volume_ratio:.1f}x\n"
                f"  Sentiment: score={sent.overall_score:.2f} ({sent.signal_strength}), "
                f"{sent.news_count} articles — {sent.summary[:80]}\n"
                f"  Options: IV_rank={opts.iv_rank}, IV_regime={opts.iv_regime}, P/C={opts.put_call_ratio}\n"
                f"  Macro: regime={macro.regime}, VIX={macro.vix_level}\n"
                f"{contract_block}"
            )

        n = len(candidates)
        slots = min(available_slots, n)
        user_prompt = (
            f"{n} candidates passed all filters. Available trade slots: {slots}. "
            f"Rank ALL of them and select the best {slots} to enter now.\n\n"
            f"CANDIDATES:\n" + "\n\n".join(blocks)
        )

        try:
            response = self.client.chat.completions.create(
                model=config.openai.brain_model,
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=150 * n + 200,
                temperature=0.2,
            )
            raw_decisions = json.loads(response.choices[0].message.content).get("decisions", [])
            decision_map = {d["symbol"]: d for d in raw_decisions}
        except Exception as e:
            logger.error(f"Batch brain LLM call failed: {e}")
            decision_map = {}

        output: List[Tuple[BrainDecision, dict]] = []
        valid_actions = {
            "BUY_CALL", "BUY_PUT",
            "SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "IRON_CONDOR",
            "HOLD", "SKIP",
        }

        for c in candidates:
            symbol = c["symbol"]
            tech = c["technical"]
            sent = c["sentiment"]
            macro = c["macro"]
            opts = c["options_flow"]

            tech_score = c.get("directional_score") or tech.technical_score
            sent_score_norm = (sent.overall_score + 1.0) / 2.0
            macro_score = macro.regime_score
            opts_score = opts.options_score
            w_tech = self.cfg.weight_technical
            w_sent = self.cfg.weight_sentiment
            w_macro = self.cfg.weight_macro
            w_opts = self.cfg.weight_options_flow
            w_total = w_tech + w_sent + w_macro + w_opts
            composite = round(
                tech_score * (w_tech / w_total) +
                sent_score_norm * (w_sent / w_total) +
                macro_score * (w_macro / w_total) +
                opts_score * (w_opts / w_total), 3
            )

            raw = decision_map.get(symbol, {})
            action = raw.get("action", "SKIP").upper()
            if action not in valid_actions:
                action = "SKIP"
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            reasoning = raw.get("reasoning", "Not selected in batch ranking.")
            target_dte = max(self.opts.min_dte, min(self.opts.max_dte, int(raw.get("target_dte", 35))))
            pos_size = max(0.03, min(config.risk.max_position_pct, float(raw.get("position_size_pct", 0.06))))
            pos_size = round(min(pos_size * macro.position_size_multiplier, config.risk.max_position_pct), 4)
            catalyst_type = raw.get("catalyst_type", "momentum")
            if catalyst_type not in {"fundamental", "momentum", "sentiment_buzz"}:
                catalyst_type = "momentum"
            shelf_life = max(3, min(15, int(raw.get("sentiment_shelf_life_days", 7))))

            score_breakdown = {
                "technical": {"score": round(tech_score, 3), "weight": w_tech,
                               "contribution": round(tech_score * (w_tech / w_total), 3)},
                "sentiment": {"score": round(sent_score_norm, 3), "weight": w_sent,
                              "contribution": round(sent_score_norm * (w_sent / w_total), 3)},
                "macro": {"score": round(macro_score, 3), "weight": w_macro,
                          "contribution": round(macro_score * (w_macro / w_total), 3)},
                "options_flow": {"score": round(opts_score, 3), "weight": w_opts,
                                 "contribution": round(opts_score * (w_opts / w_total), 3)},
                "composite": composite,
                "batch_ranked": True,
            }

            logger.info(f"Batch → {symbol}: {action} (conf={confidence:.2f}, composite={composite:.3f})")

            output.append((BrainDecision(
                symbol=symbol,
                action=action,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                technical_score=tech_score,
                sentiment_score_normalized=round(sent_score_norm, 3),
                macro_score=round(macro_score, 3),
                options_flow_score=round(opts_score, 3),
                composite_score=composite,
                score_breakdown=score_breakdown,
                strategy_mode=action,
                target_dte=target_dte,
                position_size_pct=pos_size,
                stop_loss_pct=self.opts.stop_loss_pct,
                profit_target_pct=self.opts.profit_target_long,
                catalyst_type=catalyst_type,
                sentiment_shelf_life_days=shelf_life,
                sentiment_score_at_entry=round(sent.overall_score, 3),
            ), c))

        # Entry actions first (sorted by confidence desc), then HOLD/SKIP
        output.sort(key=lambda x: (0 if x[0].action in ENTRY_ACTIONS else 1, -x[0].confidence))
        return output
