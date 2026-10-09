# Design Brief — "Trend-Buyer v1": a long-options trend-following bot ($25k paper)

*Planning doc, 2026-06-27. Straight calls/puts, no spreads. Grounded in two live reference bots
(v2 EMA-crossover client bot; the abandoned run-1 AI bot) + the [STRATEGY_EVIDENCE_LEDGER](../STRATEGY_EVIDENCE_LEDGER.md)
+ external research (trend-following, options strike/DTE, exits, sizing). Nothing built yet.*

Plain-English note: an "option" bets a stock goes up (a **call**) or down (a **put**). When you BUY one,
the most you can lose is what you paid (the "premium"). The buyer's enemy is **theta** — time decay — the
option loses a little value every day just from time passing, like a melting ice cube. This whole design
is built to beat that melt.

---

## 0. Decisions LOCKED (2026-06-27)

These six forks from §6 are now decided and drive the v1 build:

1. **Pricey-IV handling → buy DEEPER in-the-money** (option 1c). When implied volatility is high, step the
   strike deeper ITM so there's almost no extrinsic value to overpay for. Neutralizes the IV problem.
2. **Partial profit-taking → ON.** Scale out part of the position at a profit target for a steadier, calmer
   equity curve, then trail the remainder. Accepted trade-off: caps the rare monster winner (a partial-sold
   BAC would not be +254%). Steady gains preferred over occasional home runs.
3. **Exit style → TRAILING stop** (not fixed target). The exact trailing algorithm is being researched
   ("find the best") — see [TRAILING_STOP_CHOICE.md](TRAILING_STOP_CHOICE.md) when complete.
4. **Signal → keep 9/21 EMA** to start, for a clean A/B comparison against the v2 client bot.
5. **Mirror the v2 client bot closely** (same universe + 9/21 signal) but ADD our filters/IV/exits, so we
   can measure whether our additions actually improve on the pure-mechanical baseline.
6. **v1 is fully MECHANICAL** (no AI/LLM). Rationale: the proven wins (BAC) and v2's survival were both
   simple mechanical rules left alone. **BUT — AI is a planned Phase 2 (explicitly wanted):** add
   an LLM layer that acts like a seasoned human trader (reading context rules can't — earnings, news,
   "this trend looks exhausted") as a confirm/veto/sizing overlay, **only after the mechanical edge is
   proven over ~20-30 honestly-filled closed bets** so we can measure exactly what the AI adds. This is a
   first-class roadmap item, not a maybe.

---

## 1. The edge thesis

**In one sentence:** Buy a stock's direction only when a real trend is confirmed, pay as little ice-cube
cost as possible (buy deep, longer-dated, when the option is cheap), then hold the few big winners as long
as the trend lasts while quickly killing the many small losers when the trend breaks.

**Four pillars:**
1. **Trend is a real, durable edge.** Trend-following/momentum has paid in every decade for ~140 years
   across dozens of markets (AQR "A Century of Evidence"; Moskowitz-Ooi-Pedersen Sharpe ~1.3), and momentum
   exists in equity options specifically (Alpha Architect). Our v2 EMA bot is a live demo: it sat flat,
   then caught the 06-24/26 drop and spiked +25%. The signal works *if you actually wait for the trend.*
2. **A crossover alone is NOT the edge — the filters are.** A raw 9/21 EMA crossover wins only ~32-50% and
   bleeds in choppy markets ("whipsaws"). Stacking filters (ADX trend-strength > 25, price on the right
   side of the 200-day, volume) lifts win-rate into the ~60-68% range. The cross is the trigger; the
   confirmation is the edge.
3. **Buying premium only pays when you buy it cheap and deep.** Implied volatility ("IV") exceeds realized
   ~85% of the time, so buyers overpay on average — our ledger confirms it (long calls lost -$3,155 and
   -$1,524 across two builds). You flip a losing game into a winning one by (a) only buying when IV is
   genuinely cheap (low IV-rank), and (b) buying deep in-the-money (delta ~0.70-0.80) — mostly real value,
   little ice cube to melt.
4. **The money is in the exits.** Trend-following is positive-skew: most trades are small losers, a few are
   giant winners that carry everything. The exit's #1 job is to not cut the rare monster short.

### The crux: "buying calls failed" vs. BAC ran +254% — same lesson stated twice

| | What FAILED (-$3,155 / -$1,524 / SOFI) | What WON (+254% BAC) |
|---|---|---|
| Strike | ATM/near-money (delta ~0.50) — max melt | ITM-ish, low breakeven (2-2.5% move) |
| DTE | Short-dated — steepest decay zone | Longer-dated, away from the red zone |
| IV (price paid) | Bought expensive IV (JPM 72%, ARKK 88-95%) | Bought cheap IV — paid little |
| Trend | SOFI extended (RSI 63), needed a 7% move | Above rising 50-day EMA, calm-bull |
| **Exit** | Tight exits sold winners **within days** | Bot was OFF → it **held**. +254% exists only because nothing cut it |

**Verdict:** losses weren't from "buying calls" — they were from buying the *wrong* calls (short-dated ATM
lottery tickets into expensive IV) and then strangling the good ones with premature theta-panic exits. BAC
proves the opposite recipe (trend-aligned + cheap IV + deeper/longer + let it run). This bot is engineered
to be BAC, repeatably.

**Honest tension:** pillars 1-2 (trend) want you to act on the cross, but pillar 3 (cheap IV) conflicts —
fresh strong trends often arrive WITH an IV spike. That fork is Open Question #1. Also: the EMA-cross
backtests are mostly crypto/forex/futures; the *principle* is solid, the exact 9/21 parameters are not
proven on US single-names.

---

## 2. Entry rules (ALL must be true)

- **A. Trigger:** daily chart only. 9 EMA crosses **above** 21 → candidate CALL; crosses **below** → candidate PUT.
- **B. Regime filter:** CALLS only when price is above a *rising* 200-day and 50-day > 200-day; PUTS the mirror.
  Skip new bets when VIX > ~28-30 and the market is bouncing off a recent low (momentum-crash zone).
- **C. Trend-strength:** ADX(14) > 25 (a 0-100 trend meter; <20 = choppy → skip). Best-documented single filter.
- **D. Volume:** crossover-day volume > 20-day average (toggle; MED confidence).
- **E. IV cheapness:** only buy when IV-rank < ~35-50. Compute IV-rank ONCE/day from daily closes (drop
  today's partial bar); sanity-bound IV/HV (reject <0.2 or >3) before it can act. (Hard ledger lessons.)
- **F. Anti-chase:** skip if price is already > ~1.25× daily-ATR extended from the 50-day.
- **G. Strike:** default **delta 0.70-0.80 (deep ITM)**. Never default to ATM/OTM (those are the losing setups).
- **H. DTE:** **60-120 days** at entry; never < 45. (LEAPS optional later.)
- **I. Liquidity & names:** most-liquid large-caps + major ETFs. OI ≥ 50; reject spread wider than ~10-15%
  of mid; size off the ask (or max(ask,mid)), never the optimistic mid.

---

## 3. Exit rules (thread the needle: not run-1's "cut too early," not v2's "never take profit")

All exits judged on the **underlying stock**, never the option's price (option marks false-fire stops).

- **Layer 1 — hard initial stop (kills small losers):** exit if the underlying closes against you ~1.5-2×
  ATR from entry, or closes back on the wrong side of the 21 EMA. Day-1 capable (fixes ARKK -$1,235).
  Backstop: option premium -50% (last resort). Guardrails: ~5-min warmup + 2 confirming cycles before acting.
- **Layer 2 — let winners run (captures the +250%):** trend-invalidation trail — stay in while price holds
  above the 21 EMA (close-based), exit on a close below; OR an ATR Chandelier trail (highest-high − 3×ATR).
- **Layer 3 — partial scaling (steadier equity):** sell ~33-50% at +75-100% on the option, move the rest's
  stop to breakeven, trail the remainder. *Honest flag: partials reduce total return on the monster trades;
  this is the #1 thing to A/B test.*
- **Layer 4 — theta cap:** close before ~21-30 DTE regardless, or roll up to a longer expiry if the trend
  is intact. Never hold into the final weeks hoping.

---

## 4. Sizing & portfolio ($25k)

- **Risk per trade:** 1% = $250 (floor 0.5% = $125 for the first ~20-30 validation trades). For a bought
  option, risk = the premium paid.
- **Contracts = (account × risk%) ÷ premium per contract.** Deep-ITM costs more → naturally small counts.
- **Hard per-position cap:** reject if even 1 contract exceeds ~$1,250 (5% of equity). Do NOT floor to 1
  contract (that bug drove run-1 to -24.7%).
- **Concurrent positions:** 5-8. **Direction cap:** ≤ ~60% of open risk pointed the same way (track net
  beta-weighted delta). **Sector cap:** max 2-3 per sector; correlation > 0.80 = "same bet."
- **Cash reserve:** ~10%.
- **Payoff-geometry gate (launch blocker):** confirm from real closed trades that avg win ≥ avg loss before
  trusting the edge (run-3 had $65 wins vs $121 losses → needed 65% just to break even).

---

## 5. Execution & measurement guardrails (non-negotiable — the ledger's meta-lesson)

1. Limit orders only, never market; cancel + skip rather than chase a bad fill.
2. Re-validate every fill against the limit floor; reject/flag phantom fills (the paper engine booked
   below-limit fills and the "skip" never fired).
3. Derive every stop / target / P&L from the ACTUAL fill, never the quote.
4. All stops on the underlying, ~5-min warmup + 2 confirming cycles; debounce all halts the same way.
5. Size off the realized fill / the ask; OI floor ≥ 50.
6. Group by underlying in every cap gate (count bets, not legs).
7. IV-rank once/day from daily closes; sanity-bound IV/HV before it changes anything.
8. Flag any open-and-close within minutes at extreme % as a suspect fill; exclude from edge measurement.
9. Judge edge ONLY on bottom-line equity over ≥20-30 honestly-filled closed bets — never on marks.

---

## 6. Open questions (decide before building)

1. **Trend vs. cheap-IV conflict.** Strong trends often arrive with an IV spike. (a) hard IV-rank gate
   (fewer, cleaner trades), (b) soft IV penalty (size down but still trade), or (c) buy deeper-ITM when IV
   is high so there's almost no extrinsic value to overpay. *Lean: (c).*
2. **Partial profit-taking: yes or no?** Steadier equity vs. capping the rare monster. Biggest single
   driver of "steady gains vs occasional home runs." *Needs your risk appetite.*
3. **Trailing stop vs. fixed profit target.** "Let winners run" is gospel, but one large backtest found
   target + breakeven beats trailing on risk-adjusted return. *Lean: A/B test on the first 20-30 trades.*
4. **Is 9/21 EMA right for US equities?** Parameters unproven on our universe. Ship 9/21 (matches v2 for a
   clean comparison), use slower 20/50, or tune first. *Evidence thin — flagged.*
5. **Mirror the v2 client bot for a clean A/B?** Keep universe/signal close to it but ADD filters/IV/exits →
   clean read on whether our additions help.
6. **Mechanical-only first, or add the AI brain?** Ledger wins (BAC) and v2's survival were both mechanical.
   *Lean: ship v1 fully mechanical; add an LLM confirm/veto only after the mechanical edge is proven.*

---

### Honest bottom line
Strongest, best-evidenced: trend-following as a durable edge (HIGH); deep-ITM + longer-DTE + cheap-IV as the
way to make buying options pay (HIGH); the execution/measurement guardrails (HIGH, learned at real cost).
Thinnest: exact EMA parameters on US equities (MED), volume confirmation (MED), whether partial-scaling
helps or hurts (genuinely contested). Those three are what the first 20-30 paper trades should resolve.
Build a trustworthy data path first; then the strategy question becomes answerable.
