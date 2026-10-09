"""
Parameter Tuner — GPT-4o powered self-improvement engine.

Weekly:  Generates a performance report + diagnosis. No parameter changes.
Monthly: Full analysis → AI recommendations → applies changes with guardrails.

Guardrails on every change:
  - Minimum 15 trades required before any parameter moves
  - Max 15% change per parameter per cycle (prevents overcorrection)
  - Absolute min/max bounds per parameter (prevents insane values)
  - Confidence threshold: AI must be ≥ 0.72 confident to apply a change
  - Max 3 parameter changes per monthly cycle (prevent over-tuning)
  - If profit_factor > 1.6 AND win_rate > 0.62: no changes (system is working, leave it alone)

Novel features:
  - Counterfactual analysis: "what if we'd used different weights last month?"
  - Component re-weighting suggestion: ranked by discrimination power
  - Regime-aware recommendations: different advice for bull vs bear periods
  - Red-flag detection: alerts for concerning patterns even without enough data to change params
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
from openai import OpenAI

from agent.database import Database
from agent.feedback.analyzer import PerformanceAnalyzer
from agent.config import config


# ── Parameter guardrails: (absolute_min, absolute_max, max_change_pct_per_cycle) ────
PARAM_GUARDRAILS = {
    "min_signal_confidence":   (0.45, 0.75, 0.15),   # Entry threshold
    "weight_technical":        (0.25, 0.55, 0.15),
    "weight_sentiment":        (0.12, 0.40, 0.15),
    "weight_macro":            (0.08, 0.28, 0.15),
    "weight_options_flow":     (0.10, 0.38, 0.15),
    "iv_rank_cheap":           (20.0, 40.0, 0.15),    # Upper bound for "cheap IV"
    "iv_rank_expensive":       (40.0, 65.0, 0.15),   # Lower bound for "expensive IV"
    "min_premium":             (40.0, 200.0, 0.20),
    "max_premium":             (400.0, 1500.0, 0.20),
    "rsi_min":                 (28.0, 45.0, 0.12),
    "rsi_max":                 (55.0, 72.0, 0.12),
}

MIN_TRADES = 15          # Minimum closed trades required to change any parameter
MAX_CHANGES = 3          # Max parameters changed per monthly cycle
MIN_AI_CONFIDENCE = 0.72 # AI must be this confident to apply a change

# Don't tune a working system
LEAVE_ALONE_WIN_RATE = 0.62
LEAVE_ALONE_PF = 1.60


class ParameterTuner:
    def __init__(self, db: Database):
        self.db = db
        self.analyzer = PerformanceAnalyzer(db)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ─────────────────────────────────────────────────────────────────
    #  PUBLIC: WEEKLY REVIEW (diagnosis only, no changes)
    # ─────────────────────────────────────────────────────────────────

    def run_weekly_review(self) -> Dict[str, Any]:
        logger.info("📊 Running weekly performance review...")
        analysis = self.analyzer.run_full_analysis(days=7)

        if analysis.get("error"):
            logger.info(f"Weekly review: {analysis['message']}")
            report = {
                "review_type": "weekly",
                "generated_at": datetime.utcnow().isoformat(),
                "status": "insufficient_data",
                "message": analysis.get("message", ""),
                "trades_analyzed": analysis.get("trades", 0),
            }
            self._save_report(report)
            return report

        report = self._build_report(analysis, mode="weekly", apply_changes=False)
        self._save_report(report)
        logger.info(f"Weekly review complete: {report.get('ai_assessment', '')[:100]}")
        return report

    # ─────────────────────────────────────────────────────────────────
    #  PUBLIC: MONTHLY REVIEW (full analysis + optional parameter changes)
    # ─────────────────────────────────────────────────────────────────

    def run_monthly_review(self) -> Dict[str, Any]:
        logger.info("🔄 Running monthly parameter review...")
        analysis = self.analyzer.run_full_analysis(days=30)

        if analysis.get("error"):
            logger.info(f"Monthly review: {analysis['message']}")
            report = {
                "review_type": "monthly",
                "generated_at": datetime.utcnow().isoformat(),
                "status": "insufficient_data",
                "message": analysis.get("message", ""),
            }
            self._save_report(report)
            return report

        # Check if system is performing well enough to leave alone
        overview = analysis.get("overview", {})
        win_rate = overview.get("win_rate", 0)
        pf = overview.get("profit_factor") or 0

        if win_rate >= LEAVE_ALONE_WIN_RATE and pf >= LEAVE_ALONE_PF:
            logger.info(
                f"✅ System performing well (WR={win_rate:.1%}, PF={pf:.2f}) — "
                f"skipping parameter changes this cycle"
            )
            report = self._build_report(analysis, mode="monthly", apply_changes=False)
            report["skip_reason"] = f"System healthy (win_rate={win_rate:.1%}, profit_factor={pf:.2f})"
            self._save_report(report)
            return report

        report = self._build_report(analysis, mode="monthly", apply_changes=True)
        self._save_report(report)

        n_changed = len(report.get("applied_changes", []))
        if n_changed > 0:
            logger.info(f"✅ Monthly review: {n_changed} parameter(s) updated")
            for change in report["applied_changes"]:
                logger.info(
                    f"   {change['parameter']}: {change['old_value']} → {change['new_value']} "
                    f"(confidence={change['confidence']:.0%})"
                )
        else:
            logger.info("✅ Monthly review complete — no changes warranted by data")

        return report

    # ─────────────────────────────────────────────────────────────────
    #  CORE: BUILD REPORT (calls GPT-4o)
    # ─────────────────────────────────────────────────────────────────

    def _build_report(self, analysis: Dict, mode: str, apply_changes: bool) -> Dict[str, Any]:
        current_params = self._read_current_params()
        prompt = self._build_prompt(analysis, current_params, mode)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the chief quantitative analyst for an AI-powered options trading system. "
                            "You review live trading performance data and recommend specific, evidence-based "
                            "parameter adjustments. You are rigorous, conservative, and data-driven. "
                            "You NEVER recommend changes without clear statistical evidence. "
                            "You are aware that over-tuning destroys systems — when in doubt, do nothing. "
                            "You always output valid JSON matching the exact schema requested."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.15,
            )
            ai = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"GPT-4o feedback review failed: {e}")
            ai = {
                "assessment": f"AI review failed: {e}",
                "top_issues": [],
                "recommendations": [],
                "red_flags": [f"Review call failed: {e}"],
                "next_cycle_focus": "Investigate API connectivity",
            }

        report: Dict[str, Any] = {
            "review_type": mode,
            "generated_at": datetime.utcnow().isoformat(),
            "period_days": analysis.get("period_days", 30),
            "total_trades_analyzed": analysis.get("overview", {}).get("total_trades", 0),
            "ai_assessment": ai.get("assessment", ""),
            "top_issues": ai.get("top_issues", []),
            "recommendations": ai.get("recommendations", []),
            "red_flags": ai.get("red_flags", []),
            "next_cycle_focus": ai.get("next_cycle_focus", ""),
            "current_parameters": current_params,
            "parameters_changed": False,
            "applied_changes": [],
            "overview_snapshot": analysis.get("overview", {}),
            "alpha_decay_status": analysis.get("alpha_decay", {}).get("status", "UNKNOWN"),
        }

        if apply_changes and report["recommendations"]:
            applied = self._apply_recommendations(
                report["recommendations"],
                analysis.get("overview", {}).get("total_trades", 0),
            )
            report["parameters_changed"] = len(applied) > 0
            report["applied_changes"] = applied

        return report

    # ─────────────────────────────────────────────────────────────────
    #  PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────────

    def _build_prompt(self, analysis: Dict, params: Dict, mode: str) -> str:
        days = analysis.get("period_days", 30)
        overview = analysis.get("overview", {})
        attribution = analysis.get("component_attribution", {})
        ranking = attribution.get("_ranking", [])

        return f"""You are reviewing {days} days of LIVE options trading performance for an autonomous AI agent.

## CURRENT SYSTEM PARAMETERS
{json.dumps(params, indent=2)}

## PERFORMANCE OVERVIEW
{json.dumps(overview, indent=2)}

## SCORE BUCKET ANALYSIS (win rate per composite score range)
{json.dumps(analysis.get('score_bucket_analysis', {}), indent=2)}

## STRATEGY BREAKDOWN (by trade type)
{json.dumps(analysis.get('strategy_breakdown', {}), indent=2)}

## IV RANK ANALYSIS (validates directional vs spread cutoff)
{json.dumps(analysis.get('iv_rank_analysis', {}), indent=2)}

## EXIT QUALITY (by exit reason)
{json.dumps(analysis.get('exit_quality', {}), indent=2)}

## SIGNAL COMPONENT ATTRIBUTION (discrimination power per component)
{json.dumps(attribution, indent=2)}
Component ranking by predictive power: {ranking}

## CONFIDENCE CALIBRATION
{json.dumps(analysis.get('confidence_calibration', {}), indent=2)}

## MARKET REGIME PERFORMANCE
{json.dumps(analysis.get('regime_performance', {}), indent=2)}

## ALPHA DECAY TREND
{json.dumps(analysis.get('alpha_decay', {}), indent=2)}

## SKIP ANALYSIS
{json.dumps(analysis.get('skip_analysis', {}), indent=2)}

## DTE & HOLDING PERIOD ANALYSIS
{json.dumps(analysis.get('holding_period_analysis', {}), indent=2)}
{json.dumps(analysis.get('dte_entry_analysis', {}), indent=2)}

---

## YOUR TASK: {"DIAGNOSE ONLY (weekly — no parameter changes)" if mode == "weekly" else "FULL ANALYSIS + RECOMMENDATIONS"}

Respond with a JSON object matching this EXACT schema:

{{
  "assessment": "2-3 sentence executive summary of what the data shows",
  "top_issues": ["specific issue 1", "specific issue 2"],
  "recommendations": [
    {{
      "parameter": "exact_parameter_name_from_list_below",
      "current_value": <number>,
      "recommended_value": <number>,
      "confidence": <0.0-1.0>,
      "reasoning": "specific data points that justify this change",
      "statistical_basis": "e.g. '23 trades in bucket X showed 71% win rate vs 48% in bucket Y'"
    }}
  ],
  "red_flags": ["any warnings, anomalies, or concerning patterns even without enough data to act"],
  "next_cycle_focus": "what to specifically watch in the next {days} days"
}}

## AVAILABLE PARAMETERS
{json.dumps(list(PARAM_GUARDRAILS.keys()), indent=2)}

## STRICT RULES
1. Minimum {MIN_TRADES} closed trades required. If fewer, return empty recommendations with a note in red_flags.
2. Only recommend changes with clear statistical evidence. Explain the specific numbers.
3. Maximum 3 recommendations total.
4. If profit_factor > 1.6 AND win_rate > 62%, return ZERO recommendations — a working system must not be tuned.
5. If alpha_decay status is IMPROVING, return ZERO recommendations — do not interrupt a positive trend.
6. If component attribution shows the current weight ordering matches predictive power ranking, do not change weights.
7. Never recommend a weight change that would make weights no longer sum to 1.0.
8. Be specific about confidence scores — only exceed 0.72 when data strongly supports the change.
9. {"FOR WEEKLY REVIEW: Diagnose only. Leave recommendations array EMPTY." if mode == "weekly" else "For monthly review: recommend changes only if evidence is compelling."}"""

    # ─────────────────────────────────────────────────────────────────
    #  APPLY RECOMMENDATIONS WITH GUARDRAILS
    # ─────────────────────────────────────────────────────────────────

    def _apply_recommendations(self, recommendations: List[Dict], trade_count: int) -> List[Dict]:
        if trade_count < MIN_TRADES:
            logger.info(f"Skipping all changes — only {trade_count} trades (need {MIN_TRADES})")
            return []

        applied = []
        for rec in recommendations[:MAX_CHANGES]:
            param = rec.get("parameter", "")
            recommended = rec.get("recommended_value")
            confidence = rec.get("confidence", 0.0)
            current = rec.get("current_value")
            reasoning = rec.get("reasoning", "")

            if not param or recommended is None or current is None:
                continue
            if param not in PARAM_GUARDRAILS:
                logger.warning(f"Ignoring unknown parameter: {param}")
                continue
            if confidence < MIN_AI_CONFIDENCE:
                logger.info(f"Skipping {param}: confidence {confidence:.2f} < {MIN_AI_CONFIDENCE}")
                continue

            abs_min, abs_max, max_pct = PARAM_GUARDRAILS[param]

            # Clamp to max change per cycle
            max_delta = abs(float(current)) * max_pct
            if abs(float(recommended) - float(current)) > max_delta:
                clamped = float(current) + (max_delta if recommended > current else -max_delta)
                logger.info(f"Clamped {param}: {recommended} → {clamped:.4f} (max {max_pct:.0%} change)")
                recommended = clamped

            # Clamp to absolute bounds
            recommended = max(abs_min, min(abs_max, float(recommended)))

            # Round appropriately
            if isinstance(current, int) or param in ("rsi_min", "rsi_max", "min_premium", "max_premium"):
                recommended = round(recommended, 1)
            else:
                recommended = round(recommended, 4)

            if recommended == float(current):
                logger.info(f"Skipping {param}: value unchanged after guardrails")
                continue

            self._write_param(param, recommended)
            self._record_change(param, current, recommended, reasoning)

            applied.append({
                "parameter": param,
                "old_value": current,
                "new_value": recommended,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        return applied

    # ─────────────────────────────────────────────────────────────────
    #  PARAMETER READ / WRITE
    # ─────────────────────────────────────────────────────────────────

    def _read_current_params(self) -> Dict:
        risk = self.db.get_risk_settings()
        s = config.strategy
        o = config.options if hasattr(config, "options") else None
        return {
            "min_signal_confidence": getattr(s, "min_signal_confidence", 0.55),
            "weight_technical":      getattr(s, "weight_technical", 0.40),
            "weight_sentiment":      getattr(s, "weight_sentiment", 0.25),
            "weight_macro":          getattr(s, "weight_macro", 0.15),
            "weight_options_flow":   getattr(s, "weight_options_flow", 0.20),
            "iv_rank_cheap":         getattr(o, "iv_rank_cheap", 30.0) if o else 30.0,
            "iv_rank_expensive":     getattr(o, "iv_rank_expensive", 50.0) if o else 50.0,
            "min_premium":           risk.min_premium if risk else 75.0,
            "max_premium":           risk.max_premium if risk else 800.0,
            "rsi_min":               getattr(s, "rsi_min", 35.0),
            "rsi_max":               getattr(s, "rsi_max", 65.0),
        }

    def _write_param(self, param: str, value) -> None:
        """Apply the parameter change. Risk settings hit DB immediately; config params need restart."""
        if param in ("min_premium", "max_premium"):
            self.db.save_risk_settings(**{param: value})
            logger.debug(f"DB updated: {param} = {value}")
        else:
            # Store in ParameterSnapshot — picked up at next scheduler restart
            logger.debug(f"Config param staged: {param} = {value} (takes effect on restart)")

    def _record_change(self, param: str, old_val, new_val, reasoning: str) -> None:
        try:
            with self.db.get_session() as session:
                from agent.database import ParameterSnapshot
                snap = ParameterSnapshot(
                    parameter_name=param,
                    old_value=str(old_val),
                    new_value=str(new_val),
                    reasoning=reasoning[:500],
                    changed_at=datetime.utcnow(),
                )
                session.add(snap)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to record parameter change: {e}")

    def _save_report(self, report: Dict) -> None:
        try:
            with self.db.get_session() as session:
                from agent.database import FeedbackReport
                rec = FeedbackReport(
                    review_type=report.get("review_type", "unknown"),
                    period_days=report.get("period_days", 0),
                    total_trades=report.get("total_trades_analyzed", 0),
                    assessment=report.get("ai_assessment", ""),
                    recommendations_json=json.dumps(report.get("recommendations", [])),
                    applied_changes_json=json.dumps(report.get("applied_changes", [])),
                    red_flags_json=json.dumps(report.get("red_flags", [])),
                    parameters_changed=report.get("parameters_changed", False),
                    alpha_decay_status=report.get("alpha_decay_status", ""),
                    overview_json=json.dumps(report.get("overview_snapshot", {})),
                )
                session.add(rec)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to save feedback report: {e}")
