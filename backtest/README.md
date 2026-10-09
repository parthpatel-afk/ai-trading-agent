# Trend-Buyer — Backtesting & Research Platform

Offline simulation of the **exact same strategy brain the live bot runs**, replayed over
years of history so we can validate the edge and study parameters *now* — instead of waiting
months for the live market to hand us a real trend. Nothing here touches the live paper
account. Built product-shaped from day one: this engine is intended to become the core of the
sellable "strategy builder + backtester."

> **Why we can backtest options from stock data alone.** An option's value = intrinsic
> (how deep in-the-money) + a little time value. Our strategy deliberately buys **deep
> in-the-money**, where intrinsic dominates, so Black-Scholes — fed a reasonable volatility —
> reconstructs the price accurately. We never buy expensive historical option-price data.
> Backtest P&L is therefore **modeled, but modeled accurately for this deep-ITM style**, and
> will be spot-checked against real recent option prices in Phase 2.

---

## Status

| Phase | What | State |
|------|------|-------|
| **0** | Shared foundation: one Black-Scholes pricer, injectable config, one pure exit decision | ✅ done |
| **1** | Pull + cache ~6 yrs of history (53 optionable names, daily) | ✅ done |
| **2** | Synthetic option pricer + validation vs real recent prices | ✅ done |
| **3** | The day-by-day simulator (reuses the real entry + exit logic) → trade ledger | ✅ done |
| **4** | Parameter sweeps (vary one setting, holding the rest fixed) | ✅ done |
| **5** | Analysis + the 3D landscape (param × date-window × outcome) + robustness view | ✅ done |
| 6 | Multi-parameter grids · intraday · product dashboard/editor | later |

---

## Phase 0 — the shared foundation (in `agent/`, used live *and* here)

So the backtest tests the **real** logic, not a copy that drifts, three pieces are now shared:

- **`agent/pricing.py`** — one Black-Scholes source of truth: `bs_price`, `bs_delta`,
  `strike_for_delta` (invert a target delta → strike, for synthesizing contracts), `norm_cdf/ppf`.
  The live executor now delegates its delta math here.
- **`agent/exits.py`** — `decide_exit(...)`, the entire layered exit (time-cap → partial →
  hard stop → Chandelier trail) as **one pure function**. The live `scheduler.run_close_monitor`
  and the backtester both call it, so they can never disagree.
- **Injectable config** — `filters.evaluate(df, dir, cfg=…)`, `strategy.*(… cfg=…)`, and
  `config.clone_with({"filters.adx_min": 20, …})`. Sweeps vary one setting while holding the
  rest fixed; the live singleton is never mutated.

Tests: `python scripts/test_exit_math.py` (18) and `python scripts/test_backtest_foundation.py`
(37) — 55 green. Live behavior is unchanged (byte-for-byte decisions verified against the logs).

## Phase 1 — the data (in `backtest/`)

- **`universe.py`** — 53 liquid, optionable US names (18 = the live default watchlist, tagged;
  the rest are the selectable set for the future product), with sectors.
- **`data/store.py`** — parquet cache, **tagged by granularity** (`cache/1d/…`, `cache/1h/…`,
  `cache/1m/…`) + a `manifest.json` of coverage/provenance. Everything loads via
  `load_bars(symbol, "1d")`, so adding intraday later needs zero downstream changes.
- **`data/fetch.py`** — yfinance fetcher. Pulls **split-adjusted, dividend-UNadjusted** prices
  (the actual traded price — correct for option intrinsic, and it reproduces the live bot's
  Alpaca readings; see below). Daily is unlimited; intraday lookback limits are enforced.

### What's cached now
`2018-01-02 → 2026-07-10` daily (the ~2 extra years before 2020 are warmup for the 200-day
regime SMA + 252-day vol-rank the filters need). **53 symbols, ~107k bars.** 44 have full
history; 9 later-IPO names start at listing (ARM 2023, HOOD/COIN/IONQ 2021, PLTR/SNOW/ABNB 2020,
UBER 2019, NIO 2018) — the sim simply won't trade a name before it existed.

## Phase 2 — the synthetic option pricer (in `backtest/pricing/`)

- **`synth.py`** — `select_contract(...)` inverts Black-Scholes to a strike for a target delta
  (the mirror of how the live bot picks off the chain), snaps to a realistic strike grid, picks a
  standard monthly ~target DTE, and prices it. `reprice(...)` values it on any later day from the
  underlying's close. Vol input = the underlying's **historical volatility (HV)** — the same
  measure the entry's cheap-vol gate uses — held fixed over the trade (deep-ITM vega is small).
- **`validate.py`** — the honesty check against **real** current option chains (yfinance).

### Validation result (`python -m backtest.pricing.validate`, 8 liquid names, ~75 DTE)

| Bucket | HV-priced err (median / p90) | IV-priced err (median) |
|---|---|---|
| **Strategy zone Δ0.70–0.85** | **6.0% / 11.9%** | 4.5% |
| All deep-ITM Δ≥0.65 | 1.6% / 12.1% | 2.2% |
| Shallower Δ<0.65 | 49.5% / 100% | 11.2% |

Reading it: **IV-priced error is small (2–4.5%) → the Black-Scholes math is correct.** Deep-ITM
HV-priced error is small because those options are **intrinsic-dominated** (the price is mostly
exact arithmetic). Shallow options are ~50% off on HV — which is precisely **why the strategy
buys deep-ITM**: theta-resistant *and* cheaply/accurately modelable. Net: **backtest deep-ITM P&L
is modeled within ~6% of real market prices** in the strategy's strike zone.

*(Gotcha learned: the comparison must use a spot same-dated as the option chain. A stale spot
shows up as a constant put-call-parity offset across all strikes and wrecks the numbers — we now
pull spot+HV live in `validate.py`. Also: `2018→now` daily cache is refreshed on the current clock.)*

## Phase 3 — the simulator (in `backtest/engine/`)

- **`portfolio.py`** — the pretend $25k book: cash, one position per symbol, buys / partial
  scale-outs / full closes at modeled prices, ONE ledger row per closed trade (total P&L =
  banked partials + final proceeds − remaining cost, exactly like the live `_book_close`).
- **`simulator.py`** — walks each trading day; **reuses the live brain**: entry =
  `compute_ema_state → strategy.decide → filters.evaluate → caps → target_delta_for →
  synth.select_contract → size_position`; exit = `exits.decide_exit`. No lookahead (bars through
  day d only); exits then entries fill at day d's close with a slippage haircut.
- **`run_backtest.py`** — CLI to run one backtest and save ledger/equity/summary to `results/`.

### TRUST TEST ✅ (the reason we can believe the numbers)
Run over the live window (`run_backtest('2026-06-29','2026-07-10', record_decisions=True)`):
**0 trades**, and every fresh-cross decision matches the live logs — e.g. the sim's **07-06 SPY**
read (close 751.28, ADX **22.2**, BULL age 2) is exactly the live bot's **07-07** read
(751.27, 22.2, age 2); AAPL 23.6/23.7, TSLA 15.2 all match to ~0.1 ADX. The one-day offset is
expected (the sim acts on the signal bar the live bot reads next morning).

### First full run — default 18 names, 2020-01→2026-07 (62s)
`8 trades · win 12.5% · P&L −$4,817 (−19.3%) · maxDD −21.5% · 7 hard-stops, 1 trail`
- **The filter funnel** (3,377 fresh signals): **85% killed by ADX<25** (2,864), then regime
  (341), volume (46), anti-chase (43) → only **8** entries. The ADX gate is the master bottleneck
  — confirming both live evals at scale.
- **Not a strategy verdict — a *parameter* verdict** (n=8 is tiny). The one winner (PLTR +83% on a
  +20% move, trailed out) shows the positive-skew engine works when a real trend appears.
- **A real finding surfaced:** a fixed delta-0.75 target is only *barely* ITM in **low-vol** names
  (WMT HV 14% → strike 3.3% ITM → ~20× leverage on intrinsic → a −3% move = −49% option loss).
  "Deep by delta" ≠ "deep by moneyness." Worth a moneyness floor — a Phase-4 thing to test.
- This is exactly what the platform is for: **Phase 4 sweeps ADX (and the rest) to find settings
  that actually trade and profit; Phase 5 renders the 3D landscape.**

## Phase 4 + 5 — sweeps, the landscape, and the 3D viz

- **`sweep/`** — `sweep_1d(param, values)` runs a full backtest per value (via `config.clone_with`,
  holding everything else fixed) and returns a metrics table + a combined ledger tagged with the
  parameter value. `parameters.py` registers the tunable dials (adx_min, vol_rank_max, target_delta,
  hard_stop_atr, …) or takes any dotted config path.
- **`analysis/landscape.py`** — pivots a sweep ledger into the cube **X = parameter, Z = date-window
  (trades bucketed by entry month/quarter), Y = outcome** (total/avg P&L, # trades, win-rate).
- **`viz/surface.py`** — interactive plotly: `surface_3d` (the landscape) + `robustness_2d`
  (all-time P&L bars + trade-count line — the "stable ridge vs lucky spike" read).
- **`run_sweep.py`** — one command: sweep → landscape → HTML. Output in `results/sweep_<tag>/`.

### First landscape — the ADX gate, 2020→2026, default 18 names

| ADX gate | trades | P&L | return | win | profit factor |
|---|---|---|---|---|---|
| 15 | 75 | +$4,313 | +17.2% | 28% | 1.13 |
| **18** | 36 | **+$5,453** | **+21.8%** | 33% | **1.34** |
| 20 | 24 | +$3,276 | +13.1% | 29% | 1.27 |
| 22 | 15 | −$6,777 | −27.1% | 13% | 0.33 |
| **25 (live default)** | 8 | **−$4,817** | **−19.3%** | 12% | 0.19 |
| 28 | 3 | −$559 | −2.2% | 33% | 0.67 |
| 32 | 0 | 0 | 0% | — | — |

**The live default (ADX 25) is one of the worst settings** — it starves the bot to 8 trades and
loses 19%. The **profitable, well-populated ridge is ADX ~15–20** (all positive, profit factor
1.1–1.3, 24–75 trades). The 22–28 pockets are negative *but tiny-sample* (3–15 trades) — read the
3D surface for whether a value holds up *across quarters* (a stable ridge) vs one lucky spike; the
15–20 cluster being uniformly green is the trustworthy signal, not any single summit. Next step:
loosen ADX toward ~18–20 (and address the low-vol moneyness issue from Phase 3), re-run, then push
the best *robust* settings to the live bot.

### Faithfulness check (data + brain reproduce the live bot)
Recomputing the live **07-07** morning read from the cache:

| | close | EMA9 | EMA21 | cross/age/fresh | ADX |
|---|---|---|---|---|---|
| SPY live log | 751.27 | 743.60 | 742.07 | BULL/2/fresh | 22.2 |
| SPY from cache | 751.28 | 743.65 | 742.12 | BULL/2/fresh | **22.2** |
| TSLA live log | 419.79 | 405.26 | 404.00 | BULL/0/fresh | 15.3 |
| TSLA from cache | 419.77 | 405.44 | 404.12 | BULL/0/fresh | 15.2 |

Matches to the cent, with identical cross age / fresh flag / gate decision. (Residual few-cent
diffs = yfinance vs Alpaca IEX feed; negligible. NVDA back-adjusts to $4.92 in 2018 ≈ its 40×
cumulative splits, confirming split adjustment.)

---

## How to run

```bash
# Pull / refresh the cache (daily, full universe)
python -m backtest.data.fetch                       # skips already-cached
python -m backtest.data.fetch --force               # re-fetch all
python -m backtest.data.fetch --symbols AAPL MSFT
python -m backtest.data.fetch --granularity 1h --start 2024-01-01   # future intraday

# Inspect the universe
python backtest/universe.py
```

Data lives in `backtest/data/cache/` (git-ignored, regenerated from source).
