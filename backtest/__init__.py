"""
Trend-Buyer — Backtesting & research platform.

Offline simulation of the SAME strategy brain the live bot runs (agent.signals.filters,
agent.strategy, agent.exits), replayed over years of history so we can validate the edge
and study parameters without waiting for the live market. Nothing here touches the live
paper account. Product-shaped from day one: this engine is intended to become the core of
the sellable "strategy builder + backtester".

Layout:
  universe.py   — the selectable ~50-name optionable universe (18 default + extras)
  data/         — pull + cache historical bars (parquet, granularity-tagged; intraday-ready)
  pricing/      — synthesize + reprice deep-ITM options from the underlying (Phase 2)
  engine/       — the day-by-day simulator + pretend portfolio (Phase 3)
  sweep/        — run many parameter sets, collect trade ledgers (Phase 4)
  analysis/     — ledgers -> metrics -> the x/y/z landscape cube (Phase 5)
  viz/          — interactive 3D surfaces (Phase 5)
"""
