# Backtesting Platform — Plan & Product Vision

*Approved 2026-07-12. The immediate goal: validate Trend-Buyer v1 on years of history instead
of waiting months for the live market. The end goal: this engine becomes the core of a sellable
"trader-bot creator." Build it product-shaped from day one.*

---

## Guiding principles

1. **One brain, two homes.** The simulator reuses the LIVE bot's real code
   (`agent.signals.filters`, `agent.strategy`, `agent.exits`, `agent.pricing`). A backtest can
   never drift from what actually runs. (Phase 0 made those pieces shared + parameterizable.)
2. **Modeled, but honestly.** We have no cheap multi-year option-price data, so we *synthesize*
   option prices with Black-Scholes. This is accurate **because we buy deep in-the-money** (price
   ≈ intrinsic + a small time-value sliver). We validate the model against real recent prices and
   are explicit that P&L is modeled.
3. **Robustness over peak-hunting.** The product must steer users toward settings that are
   *stable across time* (broad plateaus), not the single luckiest past value (a lonely spike).
   Overfitting is the #1 way tools like this lie to people.
4. **Nothing here touches the live account.** Pure offline research.

---

## The price-data choice (settled in Phase 1)

Cache **split-adjusted, dividend-UNadjusted** daily prices (yfinance `auto_adjust=False`, keep
`close`):
- split-adjusted → moving averages stay continuous across splits (NVDA 40×, AMZN 20:1, …);
- dividend-UNadjusted → the *actual traded price*, which is what an option's intrinsic value
  (spot − strike) references. A dividend-adjusted series would misprice the options.
- Bonus: it reproduces the live bot's raw Alpaca readings (verified to the cent on 07-07).

Dividends are thus a small, separate real-world drag on long call holds — noted, modeled later
if it matters. Data granularity is **daily now**, but the cache is tagged by granularity so
intraday (hourly/minute) plugs in later with zero downstream rework.

---

## Phases

| Phase | Deliverable | State |
|------|-------------|-------|
| **0** | Shared foundation: `agent/pricing.py`, `agent/exits.py` (pure `decide_exit`), injectable config (`clone_with`). 55 tests. | ✅ |
| **1** | `backtest/universe.py` (53 names), `backtest/data/` (parquet cache + yfinance fetch). 2018→2026 daily, ~107k bars. | ✅ |
| **2** | Synthetic option pricer (build the deep-ITM contract via `strike_for_delta`, reprice daily via `bs_price`; HV as the vol input). **Validate vs real recent Alpaca option prices** and report the error. | ⏳ |
| **3** | The simulator: walk each trading day, run the real entry (filters+sizing) at the "open" and the real `decide_exit` at the "close", tracking a pretend $25k book → a **trade ledger** (one row per buy/sell w/ dates, P&L, params). **Trust test first:** reproduce the live 0-trade result over 06-30→07-09 (all ADX-blocked); only then run 2020→2026. | ⏳ |
| **4** | Parameter sweeps: vary one setting across a range (holding others fixed) or a full grid, via `config.clone_with`. Collect a ledger per run. Split results by calls vs puts. | ⏳ |
| **5** | Analysis + the **3D landscape**: from the ledgers build the cube — X = a parameter, Y = an outcome (total/avg P&L, # trades, win-rate), Z = date-window (month buckets, tunable 2/6-mo). Interactive plotly surface; overlay years (2020…2026); a plateau-vs-spike robustness read. | ⏳ |
| **6** | Later: intraday data + intraday strategies; the product dashboard + visual strategy editor; free/paid tiering; cloud-vs-local. | later |

---

## The 3D landscape (concretely)

Every backtest emits a **trade ledger** (every simulated buy/sell: entry date, direction,
strike, entry/exit price, contracts, P&L $ and %, plus the parameter set + entry metrics like
vol-rank/ADX). The landscape is then just a grouping/pivot on that ledger:

- **X** = the parameter under study (IV/vol cap, ADX min, delta depth, stop distance, …).
- **Y** = the chosen outcome (total P&L, avg P&L/trade, # of moves, win-rate).
- **Z** = date-window (bucket trades by entry month; window size tunable — 2-mo, 6-mo, yearly).
- **Peaks** = parameter values that produced the best outcome in that window.
- **Broad plateau** = robust (generalizes); **lonely spike** = overfit (ignore).
- **Year overlays** (2020…2026 surfaces superimposed) = separate event-driven from seasonal
  effects, and see which settings held up across regimes.

One parameter varies at a time for a clean read; full multi-parameter grids are also supported.

---

## Product vision & honest cautions (my take, for later)

The idea is sound and sellable (cf. Composer, QuantConnect, TradingView's tester — none great
for options). A strategy really is *a flowchart of rules + parameters*, which is exactly the
right thing to make user-editable. Build the engine once so the same definition runs live and in
backtest. Tiering: free = default strategy + read-only dashboard; paid = the editor + backtester
+ dynamic parameters (that's where willingness-to-pay is).

Four cautions to design around now, not later:
1. **Overfitting** — frame the whole UX around robustness (plateaus), never "find the max."
2. **Options-pricing scope** — synthetic pricing is honest for deep-ITM/daily; short-dated ATM /
   0DTE / intraday would need real (expensive) option data. Gate the builder to what we can price
   honestly, or clearly label "modeled."
3. **Data licensing** — shipping market data hits vendor limits; "user brings their own Alpaca
   key, runs locally" sidesteps most of it.
4. **Cloud vs local** — local = cheaper + better licensing + user trust; cloud = recurring
   revenue + easier UX but you carry the data/compute bill. Likely hybrid (local engine, hosted
   dashboard). Decide later; doesn't block the engine.
