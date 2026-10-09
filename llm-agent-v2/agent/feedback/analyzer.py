"""
Performance Analyzer — Deep multi-dimensional analysis of every trade, decision, and skip.

Dimensions analyzed:
  - Overall P&L, win rate, profit factor, expectancy
  - Win rate by composite score bucket (validates entry threshold)
  - Win rate by strategy type (BUY_CALL / BUY_PUT / BULL_SPREAD / BEAR_SPREAD)
  - Win rate by IV rank range (validates the <40 directional, >=40 spread rule)
  - Win rate by symbol (identifies our best/worst performers)
  - Win rate by market regime (bull/bear/sideways)
  - Exit quality (did stop losses save us? did targets cut too early?)
  - Confidence calibration (does GPT-4o confidence actually predict wins?)
  - Component attribution (which of the 4 signals is most predictive?)
  - Alpha decay (is performance improving or deteriorating over time?)
  - Skip analysis (how selective are we? what's blocking most entries?)
  - Consecutive patterns (max winning/losing streaks for drawdown context)
  - Counterfactual (shadow): what would have happened on our best skips?
"""
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

from agent.database import Database, AgentDecision, Trade


class PerformanceAnalyzer:
    def __init__(self, db: Database):
        self.db = db

    # ─────────────────────────────────────────────────────────────────
    #  MASTER ENTRY POINT
    # ─────────────────────────────────────────────────────────────────

    def run_full_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Run every sub-analysis and return a single comprehensive report dict."""
        closed = self.db.get_closed_trades(days=days)
        decisions = self._get_decisions(days=days)

        if len(closed) < 3:
            return {
                "error": "insufficient_data",
                "message": f"Only {len(closed)} closed trades in the last {days} days. Need at least 3.",
                "trades": len(closed),
                "decisions": len(decisions),
            }

        logger.info(f"Analyzing {len(closed)} closed trades + {len(decisions)} decisions over {days}d")

        decision_map = {d.trade_id: d for d in decisions if d.trade_id}

        return {
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat(),
            "overview": self._overview(closed),
            "score_bucket_analysis": self._score_bucket_analysis(closed, decision_map),
            "strategy_breakdown": self._strategy_breakdown(closed),
            "symbol_performance": self._symbol_performance(closed),
            "iv_rank_analysis": self._iv_rank_analysis(closed, decision_map),
            "exit_quality": self._exit_quality(closed),
            "confidence_calibration": self._confidence_calibration(closed, decision_map),
            "component_attribution": self._component_attribution(closed, decision_map),
            "regime_performance": self._regime_performance(closed, decision_map),
            "alpha_decay": self._alpha_decay(days=days),
            "skip_analysis": self._skip_analysis(decisions),
            "consecutive_patterns": self._consecutive_patterns(closed),
            "holding_period_analysis": self._holding_period_analysis(closed),
            "dte_entry_analysis": self._dte_entry_analysis(closed),
        }

    # ─────────────────────────────────────────────────────────────────
    #  OVERVIEW
    # ─────────────────────────────────────────────────────────────────

    def _overview(self, trades: List[Trade]) -> Dict:
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(t.pnl or 0 for t in trades)
        gross_wins = sum(t.pnl for t in wins) if wins else 0
        gross_losses = abs(sum(t.pnl for t in losses)) if losses else 0

        pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4),
            "total_pnl": round(total_pnl, 2),
            "gross_wins": round(gross_wins, 2),
            "gross_losses": round(gross_losses, 2),
            "avg_win_dollars": round(gross_wins / len(wins), 2) if wins else 0,
            "avg_loss_dollars": round(gross_losses / len(losses), 2) if losses else 0,
            "profit_factor": round(gross_wins / gross_losses, 3) if gross_losses > 0 else None,
            "expectancy_per_trade": round(total_pnl / len(trades), 2),
            "avg_pnl_pct": round(statistics.mean(pnl_pcts), 2) if pnl_pcts else None,
            "win_loss_ratio": round(
                (gross_wins / len(wins)) / (gross_losses / len(losses)), 3
            ) if wins and losses else None,
        }

    # ─────────────────────────────────────────────────────────────────
    #  SCORE BUCKET — validates entry threshold
    # ─────────────────────────────────────────────────────────────────

    def _score_bucket_analysis(self, trades: List[Trade], decision_map: Dict) -> Dict:
        """
        Splits trades by the composite score that triggered the entry.
        If the 0.55-0.65 bucket has a higher win rate than 0.65-0.75,
        the entry threshold needs re-examination.
        """
        buckets: Dict[str, List[Trade]] = {
            "0.55-0.60": [], "0.60-0.65": [], "0.65-0.70": [],
            "0.70-0.75": [], "0.75-0.80": [], "0.80-0.85": [], "0.85+": [],
        }

        for t in trades:
            d = decision_map.get(t.id)
            if not d:
                continue
            score = None
            if d.signal_summary and isinstance(d.signal_summary, dict):
                score = d.signal_summary.get("composite_score")
            if score is None:
                score = d.confidence  # fallback

            if score is None:
                continue

            if score < 0.60:   buckets["0.55-0.60"].append(t)
            elif score < 0.65: buckets["0.60-0.65"].append(t)
            elif score < 0.70: buckets["0.65-0.70"].append(t)
            elif score < 0.75: buckets["0.70-0.75"].append(t)
            elif score < 0.80: buckets["0.75-0.80"].append(t)
            elif score < 0.85: buckets["0.80-0.85"].append(t)
            else:              buckets["0.85+"].append(t)

        result = {}
        for bucket, bt in buckets.items():
            if not bt:
                result[bucket] = {"count": 0, "win_rate": None, "avg_pnl": None}
                continue
            wins = [t for t in bt if t.pnl and t.pnl > 0]
            result[bucket] = {
                "count": len(bt),
                "win_rate": round(len(wins) / len(bt), 4),
                "avg_pnl": round(sum(t.pnl or 0 for t in bt) / len(bt), 2),
                "total_pnl": round(sum(t.pnl or 0 for t in bt), 2),
            }
        return result

    # ─────────────────────────────────────────────────────────────────
    #  STRATEGY BREAKDOWN
    # ─────────────────────────────────────────────────────────────────

    def _strategy_breakdown(self, trades: List[Trade]) -> Dict:
        breakdown: Dict[str, Dict] = {}
        for t in trades:
            stype = t.strategy_type or "unknown"
            if stype not in breakdown:
                breakdown[stype] = {"count": 0, "wins": 0, "total_pnl": 0.0, "pnl_pcts": []}
            breakdown[stype]["count"] += 1
            breakdown[stype]["total_pnl"] += t.pnl or 0
            if t.pnl and t.pnl > 0:
                breakdown[stype]["wins"] += 1
            if t.pnl_pct is not None:
                breakdown[stype]["pnl_pcts"].append(t.pnl_pct)

        for stype, d in breakdown.items():
            d["win_rate"] = round(d["wins"] / d["count"], 4) if d["count"] else 0
            d["avg_pnl"] = round(d["total_pnl"] / d["count"], 2) if d["count"] else 0
            d["avg_pnl_pct"] = round(statistics.mean(d["pnl_pcts"]), 2) if d["pnl_pcts"] else None
            del d["pnl_pcts"]
        return breakdown

    # ─────────────────────────────────────────────────────────────────
    #  SYMBOL PERFORMANCE
    # ─────────────────────────────────────────────────────────────────

    def _symbol_performance(self, trades: List[Trade]) -> Dict:
        perf: Dict[str, Dict] = {}
        for t in trades:
            sym = t.symbol
            if sym not in perf:
                perf[sym] = {"count": 0, "wins": 0, "total_pnl": 0.0}
            perf[sym]["count"] += 1
            perf[sym]["total_pnl"] += t.pnl or 0
            if t.pnl and t.pnl > 0:
                perf[sym]["wins"] += 1

        for sym, d in perf.items():
            d["win_rate"] = round(d["wins"] / d["count"], 4) if d["count"] else 0
            d["avg_pnl"] = round(d["total_pnl"] / d["count"], 2) if d["count"] else 0

        return dict(sorted(perf.items(), key=lambda x: x[1]["total_pnl"], reverse=True))

    # ─────────────────────────────────────────────────────────────────
    #  IV RANK ANALYSIS — validates the <40 directional rule
    # ─────────────────────────────────────────────────────────────────

    def _iv_rank_analysis(self, trades: List[Trade], decision_map: Dict) -> Dict:
        buckets: Dict[str, List[Trade]] = {
            "<20": [], "20-30": [], "30-40": [],
            "40-50": [], "50-60": [], "60-80": [], "80+": [],
        }

        for t in trades:
            d = decision_map.get(t.id)
            if not d or not d.signal_summary:
                continue
            iv = d.signal_summary.get("iv_rank") if isinstance(d.signal_summary, dict) else None
            if iv is None:
                iv = t.iv_at_entry

            if iv is None:
                continue

            if iv < 20:        buckets["<20"].append(t)
            elif iv < 30:      buckets["20-30"].append(t)
            elif iv < 40:      buckets["30-40"].append(t)
            elif iv < 50:      buckets["40-50"].append(t)
            elif iv < 60:      buckets["50-60"].append(t)
            elif iv < 80:      buckets["60-80"].append(t)
            else:              buckets["80+"].append(t)

        result = {}
        for bucket, bt in buckets.items():
            if not bt:
                result[bucket] = {"count": 0, "win_rate": None, "avg_pnl": None}
                continue
            wins = [t for t in bt if t.pnl and t.pnl > 0]
            result[bucket] = {
                "count": len(bt),
                "win_rate": round(len(wins) / len(bt), 4),
                "avg_pnl": round(sum(t.pnl or 0 for t in bt) / len(bt), 2),
            }
        return result

    # ─────────────────────────────────────────────────────────────────
    #  EXIT QUALITY
    # ─────────────────────────────────────────────────────────────────

    def _exit_quality(self, trades: List[Trade]) -> Dict:
        """
        Breaks down performance by exit reason.
        High avg_pnl on time_exits = we should exit sooner.
        Many stop_losses with large losses = stop is too loose.
        """
        by_reason: Dict[str, List[Trade]] = {}
        for t in trades:
            reason = t.exit_reason or "unknown"
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(t)

        result = {}
        for reason, rt in by_reason.items():
            wins = [t for t in rt if t.pnl and t.pnl > 0]
            pnl_pcts = [t.pnl_pct for t in rt if t.pnl_pct is not None]
            result[reason] = {
                "count": len(rt),
                "wins": len(wins),
                "win_rate": round(len(wins) / len(rt), 4),
                "avg_pnl": round(sum(t.pnl or 0 for t in rt) / len(rt), 2),
                "total_pnl": round(sum(t.pnl or 0 for t in rt), 2),
                "avg_pnl_pct": round(statistics.mean(pnl_pcts), 2) if pnl_pcts else None,
            }
        return result

    # ─────────────────────────────────────────────────────────────────
    #  CONFIDENCE CALIBRATION
    # ─────────────────────────────────────────────────────────────────

    def _confidence_calibration(self, trades: List[Trade], decision_map: Dict) -> Dict:
        """
        Is GPT-4o's confidence score actually predictive?
        If high-confidence trades don't win more often, the model's self-assessment is broken.
        """
        buckets: Dict[str, List[Trade]] = {
            "<0.60": [], "0.60-0.70": [], "0.70-0.80": [], "0.80-0.90": [], "0.90+": [],
        }

        for t in trades:
            d = decision_map.get(t.id)
            if not d or d.confidence is None:
                continue
            c = d.confidence
            if c < 0.60:      buckets["<0.60"].append(t)
            elif c < 0.70:    buckets["0.60-0.70"].append(t)
            elif c < 0.80:    buckets["0.70-0.80"].append(t)
            elif c < 0.90:    buckets["0.80-0.90"].append(t)
            else:             buckets["0.90+"].append(t)

        result = {}
        for bucket, bt in buckets.items():
            if not bt:
                result[bucket] = {"count": 0, "win_rate": None}
                continue
            wins = [t for t in bt if t.pnl and t.pnl > 0]
            result[bucket] = {
                "count": len(bt),
                "win_rate": round(len(wins) / len(bt), 4),
                "avg_pnl": round(sum(t.pnl or 0 for t in bt) / len(bt), 2),
            }

        # Calibration score: 1.0 = perfectly calibrated (each bucket wins more than the previous)
        rates = [v["win_rate"] for v in result.values() if v["win_rate"] is not None]
        monotonic_violations = sum(1 for i in range(1, len(rates)) if rates[i] < rates[i-1])
        result["_calibration_violations"] = monotonic_violations
        result["_well_calibrated"] = monotonic_violations <= 1

        return result

    # ─────────────────────────────────────────────────────────────────
    #  COMPONENT ATTRIBUTION
    # ─────────────────────────────────────────────────────────────────

    def _component_attribution(self, trades: List[Trade], decision_map: Dict) -> Dict:
        """
        For winning vs losing trades, what were the avg component scores?
        If Technical is high on both wins and losses, it's not discriminating well.
        If Sentiment is much higher on wins than losses, it's our best predictor.
        """
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]

        def avg_component(tlist, component_key):
            vals = []
            for t in tlist:
                d = decision_map.get(t.id)
                if d and d.signal_summary and isinstance(d.signal_summary, dict):
                    v = d.signal_summary.get(component_key)
                    if v is not None:
                        vals.append(v)
            return round(statistics.mean(vals), 4) if vals else None

        components = ["technical_score", "sentiment_score", "macro_score", "options_flow_score"]

        result = {}
        for comp in components:
            win_avg = avg_component(wins, comp)
            loss_avg = avg_component(losses, comp)
            result[comp] = {
                "avg_on_wins": win_avg,
                "avg_on_losses": loss_avg,
                "discrimination": round(win_avg - loss_avg, 4) if win_avg is not None and loss_avg is not None else None,
                # Higher discrimination = better predictor
            }

        # Rank components by discriminating power
        ranked = sorted(
            [(k, v["discrimination"]) for k, v in result.items() if v["discrimination"] is not None],
            key=lambda x: x[1], reverse=True
        )
        result["_ranking"] = [k for k, _ in ranked]
        return result

    # ─────────────────────────────────────────────────────────────────
    #  REGIME PERFORMANCE
    # ─────────────────────────────────────────────────────────────────

    def _regime_performance(self, trades: List[Trade], decision_map: Dict) -> Dict:
        by_regime: Dict[str, List[Trade]] = {}
        for t in trades:
            d = decision_map.get(t.id)
            regime = "unknown"
            if d and d.signal_summary and isinstance(d.signal_summary, dict):
                regime = d.signal_summary.get("market_regime", "unknown")
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(t)

        result = {}
        for regime, rt in by_regime.items():
            wins = [t for t in rt if t.pnl and t.pnl > 0]
            result[regime] = {
                "count": len(rt),
                "win_rate": round(len(wins) / len(rt), 4),
                "avg_pnl": round(sum(t.pnl or 0 for t in rt) / len(rt), 2),
                "total_pnl": round(sum(t.pnl or 0 for t in rt), 2),
            }
        return result

    # ─────────────────────────────────────────────────────────────────
    #  ALPHA DECAY — is the system getting better or worse?
    # ─────────────────────────────────────────────────────────────────

    def _alpha_decay(self, days: int = 30) -> Dict:
        half = max(days // 2, 7)
        recent = self.db.get_closed_trades(days=half)
        all_period = self.db.get_closed_trades(days=days)

        recent_ids = {t.id for t in recent}
        older = [t for t in all_period if t.id not in recent_ids]

        def period_stats(tlist):
            if not tlist:
                return {"count": 0, "win_rate": None, "total_pnl": 0, "profit_factor": None}
            wins = [t for t in tlist if t.pnl and t.pnl > 0]
            losses = [t for t in tlist if t.pnl and t.pnl <= 0]
            gw = sum(t.pnl for t in wins) if wins else 0
            gl = abs(sum(t.pnl for t in losses)) if losses else 0
            return {
                "count": len(tlist),
                "win_rate": round(len(wins) / len(tlist), 4),
                "total_pnl": round(sum(t.pnl or 0 for t in tlist), 2),
                "profit_factor": round(gw / gl, 3) if gl > 0 else None,
            }

        recent_stats = period_stats(recent)
        older_stats = period_stats(older)

        trend = None
        if recent_stats["win_rate"] is not None and older_stats["win_rate"] is not None:
            trend = round(recent_stats["win_rate"] - older_stats["win_rate"], 4)

        return {
            f"recent_{half}d": recent_stats,
            f"prior_{half}d": older_stats,
            "win_rate_trend": trend,
            "status": (
                "IMPROVING" if trend is not None and trend > 0.05 else
                "STABLE" if trend is not None and abs(trend) <= 0.05 else
                "DECAYING" if trend is not None and trend < -0.05 else
                "INSUFFICIENT_DATA"
            ),
        }

    # ─────────────────────────────────────────────────────────────────
    #  SKIP ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def _skip_analysis(self, decisions: List[AgentDecision]) -> Dict:
        skips = [d for d in decisions if d.action in ("SKIP", "HOLD")]
        entries = [d for d in decisions if d.action not in ("SKIP", "HOLD")]
        executed = [d for d in entries if d.executed]
        blocked = [d for d in entries if not d.executed]

        blocked_reasons: Dict[str, int] = {}
        for d in blocked:
            reason = d.execution_blocked_reason or "unknown"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

        skip_symbols: Dict[str, int] = {}
        for d in skips:
            skip_symbols[d.symbol] = skip_symbols.get(d.symbol, 0) + 1

        return {
            "total_decisions": len(decisions),
            "entries_attempted": len(entries),
            "entries_executed": len(executed),
            "entries_blocked_by_risk": len(blocked),
            "skips_by_ai": len(skips),
            "selectivity_rate": round(len(executed) / len(decisions), 4) if decisions else 0,
            "top_blocked_reasons": dict(
                sorted(blocked_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "most_skipped_symbols": dict(
                sorted(skip_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }

    # ─────────────────────────────────────────────────────────────────
    #  CONSECUTIVE PATTERNS
    # ─────────────────────────────────────────────────────────────────

    def _consecutive_patterns(self, trades: List[Trade]) -> Dict:
        if not trades:
            return {}
        sorted_trades = sorted(
            trades,
            key=lambda t: t.exit_time or datetime.min
        )
        max_wins = cur_wins = 0
        max_losses = cur_losses = 0
        for t in sorted_trades:
            if t.pnl and t.pnl > 0:
                cur_wins += 1
                cur_losses = 0
                max_wins = max(max_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_losses = max(max_losses, cur_losses)

        return {
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
            "current_streak": cur_wins if cur_wins > 0 else -cur_losses,
        }

    # ─────────────────────────────────────────────────────────────────
    #  HOLDING PERIOD ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def _holding_period_analysis(self, trades: List[Trade]) -> Dict:
        """Do longer holds win more? Helps calibrate DTE targets."""
        short = [t for t in trades if t.dte_at_entry and t.dte_at_entry <= 21]
        medium = [t for t in trades if t.dte_at_entry and 22 <= t.dte_at_entry <= 35]
        long_ = [t for t in trades if t.dte_at_entry and t.dte_at_entry > 35]

        def stats(tlist):
            if not tlist:
                return {"count": 0, "win_rate": None, "avg_pnl": None}
            wins = [t for t in tlist if t.pnl and t.pnl > 0]
            return {
                "count": len(tlist),
                "win_rate": round(len(wins) / len(tlist), 4),
                "avg_pnl": round(sum(t.pnl or 0 for t in tlist) / len(tlist), 2),
            }

        return {
            "dte_<=21": stats(short),
            "dte_22-35": stats(medium),
            "dte_>35": stats(long_),
        }

    # ─────────────────────────────────────────────────────────────────
    #  DTE AT ENTRY ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def _dte_entry_analysis(self, trades: List[Trade]) -> Dict:
        """Average DTE at entry for wins vs losses."""
        wins = [t for t in trades if t.pnl and t.pnl > 0 and t.dte_at_entry]
        losses = [t for t in trades if t.pnl and t.pnl <= 0 and t.dte_at_entry]
        return {
            "avg_dte_on_wins": round(statistics.mean([t.dte_at_entry for t in wins]), 1) if wins else None,
            "avg_dte_on_losses": round(statistics.mean([t.dte_at_entry for t in losses]), 1) if losses else None,
        }

    # ─────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _get_decisions(self, days: int) -> List[AgentDecision]:
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            return session.query(AgentDecision).filter(
                AgentDecision.timestamp >= cutoff
            ).order_by(AgentDecision.timestamp.desc()).all()
