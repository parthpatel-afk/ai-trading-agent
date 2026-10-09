# Strategy Evidence Ledger — Algorithmic Options-Trading Bot

> **Provenance.** Compiled 2026-06-27 from every build/run (runs 1–3) of the prior bot, before that build was archived. Mined from all three trade databases (live + archived run-1/run-2), the full CHANGELOG/NOTES history, and 11 days of run logs, plus outside research. The run-3 build itself has been archived to `archive/run3_account-[paper-account]_2026-06-27/`.
>
> **This file is the bridge.** It is the ONE knowledge artifact kept out of the archive on purpose. It carries forward *what the evidence showed* — not what we assumed — so the next build does not relearn the same lessons at real cost. It is deliberately **strategy-neutral**: it reports what worked, failed, or remains unproven; it does **not** prescribe which strategy to run next. A machine-readable companion lives in `STRATEGY_EVIDENCE_LEDGER.json`.

---

## 1. How to read this

**The rule.** Evidence over assumption. Every claim carries data and a source. Low-sample or replay-only items are flagged as such. Nothing is overstated; suspected-but-unproven items live in the OPEN section, kept separable from proven ones.

**Label legend.**
- **Verdict** — `WORKS` (evidence supports it), `FAILS` (evidence shows it loses/breaks), `MIXED` (partly true / depends on conditions), `UNTESTED` (fix or claim is engineering-validated by replay/unit test but never confirmed against live fills).
- **Confidence** — `HIGH` (multiple independent observations or hard quantified data), `MED` (single solid observation or small sample), `LOW` (anecdotal / tiny sample).
- **Real-money?** — `yes` = observed in our own paper trades/logs/DBs (real bot behavior on a paper account). `research` = external literature, not our own trades. *("real_money" means observed-in-our-system, not live cash — the account was paper throughout.)*

**Critical caveat that colors the entire ledger.** The bot was **STOPPED 2026-06-14 with ~1 confirmed live close**. Nearly every 2026-06 code fix (limit orders, atomic close, stop warmup, bet-counting) is validated by replay/unit test **only** and is `UNTESTED` against real fills. Treat all "fixed" claims as engineering-validated, not market-proven.

---

## 2. Merged lessons (deduplicated, grouped by category, sorted by confidence then verdict)

### EXECUTION & FILLS

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **Never market-order multi-leg spreads; price a net LIMIT near mid** | MIXED→FAILS-as-built | HIGH | yes | 06-17 market MLEG fills 25–54% under mid: XLF 169→78, AMZN 120→85, NVDA 158→120, TSLA 109→80; −$508 day | Market combos cross both legs' bid-ask; on thin spreads that crossing is a huge fraction of the credit | Always net-LIMIT at ≥95% of mid; poll for the real fill; CANCEL+SKIP rather than chase; accept non-fill over bad fill |
| **Paper engine returns fills that VIOLATE the limit floor; bot books them — "skip" never fires** | FAILS | HIGH | yes | 06-25: META 117→40 (−66%), MSFT 104→35, GOOGL 109→55, AAPL 128→80, DIA 120→85; 06-22 AAPL 138→110, AMZN 114→85; "skipping to avoid a bad fill" = **0** occurrences in 11 logs | LIMIT at mid×0.95 was coded to cancel+skip, but `filled_avg_price` came back below the floor and was booked anyway | Re-validate booked credit ≥ limit×contracts AFTER poll; hard-assert it; reject/flag phantom fills; don't trust paper `filled_avg_price` |
| **Stops/P&L keyed off COLLECTED credit must use the REAL fill, never the quote** | WORKS | HIGH | yes | NVDA booked −$120 (mark) vs ~−$10 (fills); 06-17 under-mid fills tightened 2×-credit stops instantly | A fill below quoted credit silently tightens a credit-relative stop and books false P&L | Read back actual fill after every open/close; derive profit target, 2×-credit stop, and realized P&L only from it |
| **Counting LEGS not BETS halved the position cap** | WORKS | HIGH | yes | 4 live legs read as 2 bets after fix; 06-16 blocked valid XLE spread as "6/6" while holding 3 bets | A 2-leg spread counted as 2 positions; cap tripped at half capacity; recurred 3× across gates | Count BETS (group legs by underlying) in EVERY gate — scan AND both entry-execution checks |
| **Non-atomic close stranded the protective long leg naked** | UNTESTED | MED | yes | NVDA 225C long stranded ~4h on 06-12; rejection 40310000 | Closing the short first left a momentary-naked window → "uncovered" rejection | Close defined-risk spreads as ONE atomic MLEG order; mark closed only when BOTH legs confirmed flat; integration-test live |
| **Sub-2-minute "wins" are fill artifacts, not edge** | MIXED | MED | yes | Run1 BULL_SPREAD 3 winners held 0.8/1.6/1.8 min at +104/+114/+122% | Spreads cannot legitimately gain 100%+ in <2 min — same marking bug that hurt run 3 helped run 1 | Flag any open-and-close within minutes at extreme % as a suspect fill; exclude from edge measurement |

### STOPS & EXITS

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **Judge stops on the underlying, never on an illiquid option bid / opening-minute mark** | WORKS | HIGH | yes | 06-05 phantom −14.5% open vs −5% next cycle; NVDA stop tripped on $2.15 gap quote (booked −$120 vs ~−$10 real) | Option-price stops false-fire on wide spreads and junk opening auction marks | Base stops on underlying trend/ATR; warmup ~5 min; require 2 consecutive confirming cycles |
| **Stop distance = 2× COLLECTED credit amplifies any slippage into instant stop-outs** | MIXED | HIGH | yes | XLF opened 10:45 ET, stopped 10:48 ET (3 min), −$182; stop = 2× collected credit | A halved credit halves the stop distance, so positions arrive already underwater | Either guarantee good fills first, or set the stop off max-loss/width not collected credit, or add a warmup |
| **Instant stop-out loop: spreads force-closed 2–8 min after open for −100% to −233%** | FAILS | HIGH | yes | AAPL 06-26 open 08:30:45 ($75) → cut 08:38:03 (−115%, −$86); 06-17 XLF −233%, AMZN −171%, NVDA −125%; 14× "credit_stop seen <4min after open" across 5 logs | Downstream symptom of phantom thin-credit fills + opening-mark noise | Fix fills first; widen warmup; require N consecutive confirmed (non-opening) stop reads before cutting |
| **Phantom P&L: stop fired on a garbage opening mark, booked a fake loss** | UNTESTED | MED | yes | NVDA marked $215 cost-to-close vs true ~$105–145 (<$190 line); booked −$120, real ~$10–50; account actually +$27.69 | Opening marks unreliable; either warmup OR 2-cycle confirm alone would have prevented it (replay-only) | No mark-based stops first ~5 min; require 2 consecutive cycles; book P&L from fills; keep a marks-independent bottom-line equity check |
| **Structure/invalidation + ATR stops beat fixed-% stops** | WORKS | MED | yes/research | ARKK −50.9%/−$1,235 in ~1 day, days_held=1, peak==entry; flat-% caught it far too late; ~$600 saveable | Velocity needs 3 days, trailing needs a prior peak — flat-% alone is blind on day 1 | Layered exits on the underlying: invalidation, ATR chandelier off post-entry high, one-shot gut-check, flat-% as last resort |
| **Fast sub-day breakdown slipped through every time-gated stop** | UNTESTED | MED | yes | ARKK −51%/−$1,235 in <1 day, days_held=1, peak_value==premium_paid | Bought near close, never ticked up; time-gated exits need multiple days | Layered stops judged off the stock that work on day 1 with no peak; re-tune live |
| **False circuit-breaker: first-of-day opening mark read −14.5% and halted trading** | WORKS | MED | yes | 06-05 open −14.5%, next cycle −5%; fix requires 2 consecutive breaches | Same opening-mark noise as the phantom-loss stop; corroborated across 2 incidents | Debounce ALL portfolio halts/stops by ≥2 consecutive readings |

### STRATEGY-EDGE / MARKET-PRINCIPLE

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **Buying short-dated ATM/near-money momentum calls = textbook losing setup (no edge)** | FAILS | HIGH | yes | Run2 BUY_CALL 0W/4L −$3,155 (premiums $1,110–$2,425, all hit stop); Run1 BUY_CALL 3W/6L −$1,524; buyers win ~25–40% vs sellers ~65–90% | Long calls lost across TWO independent builds; theta + 50% stop guarantees frequent full-fraction losses | Do not rebuild a long-only buy-ATM engine; if buying, ITM/longer-dated in cheap IV with a catalyst, sized small |
| **Option BUYERS win ~25–40%; SELLERS ~65–90% — structural asymmetry** | WORKS | HIGH | yes/research | Our buy-only record matched theory: run-2 1W/4L −$2,028, run-1 6/14 +$504 (~breakeven) | Retail's losing crowd is net-long calls; asymmetry is structural | Default posture favors being the seller for base win-rate; buying is the exception requiring justification |
| **Volatility risk premium: IV > realized ~85% of the time — only the SELLER collects** | WORKS | HIGH | research | IV > realized ~85% on indices | The VRP is a real persistent edge paid only to the seller; the buyer is the structural overpayer | Harvest VRP when IV rich (elevated IV-rank AND IV/HV > ~1.2); buy only when IV genuinely cheap |
| **Strike asymmetry: OTM worst-to-buy/best-to-sell; ITM (Δ0.70+) best-to-buy** | WORKS | HIGH | research | OTM buy win ~10–40%; ITM buy win ~60–90% | Win-rate by strike is inverted for buyers vs sellers | Never naked-buy OTM; for rare directional buys use ITM Δ0.70+; sell OTM as the short leg of credit structures |
| **Directional concentration: 2:1 bearish short-call skew into a rally drove 79% loss rate** | FAILS | HIGH | yes | 14 SELL_CALL_SPREAD vs 7 SELL_PUT_SPREAD; 06-25 up-day 4 call vs 1 put; exits 15L/4W (79% loss) | Selling call spreads kept fighting a rising tape and got run over | Cap net directional exposure; add a trend/regime filter that blocks selling call spreads while price makes new highs |
| **SELL_CALL_SPREAD bled in run 3; SELL_PUT_SPREAD nearly breakeven** | MIXED | MED | yes | Run3 SELL_CALL_SPREAD 12 trades 1W/11L −$1,226; SELL_PUT_SPREAD 7 trades 3W/4L −$328 | The call side systematically faded an uptrend; instant-stop phantom fills also concentrated there | Add a regime/trend filter before selling call spreads; bias toward put-spread side until a regime signal confirms |
| **Expensive IV ignored at strategy selection — bought calls into IV-rank >70%** | MIXED | MED | yes | JPM 72.4% (−20.6%), SPY 61.5%, ARKK 88–95% still got BUY_CALL | IV was computed but never used in strategy choice; superseded by the sell-side redesign (itself unproven) | Don't BUY options when IV rich; require BOTH high IV-rank AND IV/HV > ~1.15–1.2 before calling vol "rich" |

### RISK & SIZING

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **Oversized position: one TSLA call at 27.6% of equity via bypassed cap drove run-1 −24.7%** | WORKS(-fixed) | HIGH | yes | TSLA $2,375/contract vs $800 cap; ARKK OI=9, mid $3.53 filled $4.85 (+37%), $2,425 vs ~$2,000 cap | A `max(1, contracts)` floor forced 1 contract through even over the dollar cap | REJECT outright if 1 contract exceeds the dollar cap; size vs max(ask,mid); OI floor ≥50 |
| **Risk engine approves on quoted mid credit, then fills 50–66% lower — stops set on phantom credit** | FAILS | HIGH | yes | META approved credit=$117 (11:30:36) → executed credit=$40 (11:30:37); 2×-credit stop only $80 | Entry risk check approves on the locked/quoted credit one second before a far smaller fill | Only approve AFTER real fill known, or re-run risk on actual credit and auto-flatten if realized c/w falls below threshold |
| **Run-3 payoff geometry negative: ~$65 capped wins vs ~$121 losses** | FAILS | HIGH | yes | avg_win $64.75 vs avg_loss −$120.87, 21.1% win rate, total −$1,554; need ~65% win-rate to break even | Wins capped at 50–59% credit-capture while the stop let losses run to ~2× credit | Rebalance exits so avg win ≥ avg loss; compute required win-rate from actual avg win/loss and design exits to make geometry positive |
| **Daily-loss limit booked a 5-day −$1,100 loss as "today"; oversized halt** | WORKS(-fixed) | MED | yes | TSLA −$1,100 over 5 days booked as one day (−11%); real day move −$662 (−8.09%) | Old logic summed lifetime trades closing today instead of mark-to-market day change | daily_pnl = equity − last_equity vs start-of-day equity; sequence P&L read AFTER same-cycle exits |
| **One-sided pile-up + broken IVR/IV-HV gauge; correlation cap explicitly skipped** | MIXED | MED | yes | All EOD bets bullish; 7/8 "up" 06-12; IVR≥50 & IV/HV≥1.2 & credit≥33% gate → 0 trades day1 then loosened repeatedly | All-bullish = one market-up bet; cap skipped live; at 0.20 delta c/w only 11–16% in low IV | Add net-delta + per-cluster correlation cap; don't co-tune IV and c/w gates on a low-IV tape; confirm c/w achievable in rich IV first |

### SIGNALS & INDICATORS

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **IV Rank from live option mid swung 7%→58% in 90 min → phantom rejections** | WORKS(-fixed) | HIGH | yes | HOOD toggled 7%↔58% intraday, stabilized 62.5% after fix; XLF correctly 16.5% cheap | IV Rank computed from live ATM mid fluctuates every second | Compute IV Rank once/day from daily-close HV; drop today's partial bar (`hist.iloc[:-1]`) |
| **IV/HV swings wildly within minutes, flipping trade DIRECTION on bad data** | FAILS | HIGH | yes | AAPL 06-25 IV/HV 0.13(08:45)→1.67(09:00), IV_Rank frozen 57.9, flipped SKIP→SELL_CALL_SPREAD; META 06-18 0.01→0.02→0.74→0.86 | HV denominator frequently broken (0.0/0.01/None reads); a single bad snapshot greenlit the entry | Sanity-bound and smooth IV/HV (reject <0.2 or >3) before it can change Mode; never enter on a value that just crossed threshold; fix HV computation |
| **Phantom equity marks trigger spurious risk halts** | FAILS | HIGH | yes | 06-18 false "Down $4319 today (−17.6%)" vs $18,846 opening equity, logged UNCONFIRMED | First post-open option marks mismark the whole book by thousands | Require confirmed marks (2+ consecutive, past warmup) for portfolio-health halts too, not just per-position stops |
| **News sentiment measures current tone, NOT next-day price; fresh feed avoids stale-cache mistiming** | WORKS | MED | yes | 06-05 pre-market SPY sentiment +0.214 ahead of a down day; AAPL gated by 2h-old score from 2 articles | Sentiment is a confirm/veto filter, not a predictor; cache TTL drove a mistimed chase | Use sentiment as same-direction confirm/veto; refresh on a news-content fingerprint, not a blind timer |
| **Cache-expiry-driven entry: stale sentiment delayed BUY then chased the high** | UNTESTED | MED | yes | AAPL cached −0.25 (5 SKIP cycles) → +0.125 at 10:45 → bought ATM $310C at local high, gave back ~$150 | Entry timing dictated by cache TTL, not a market event | Fingerprint-keyed sentiment refresh + ATR-normalized anti-chase guard; validate live |

### MARKET-PRINCIPLE (theta / DTE — research, strategy-neutral)

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **Theta decay accelerates non-linearly toward expiry (quantified %/day)** | WORKS | HIGH | research | ~3.3%/day @45DTE, ~6.7% @30, ~11.1% @21, 10–25% @7–14DTE; ~50% of time value lost in final 30 days | Decay roughly doubles each step toward expiry; it just flips sign for buyer vs seller | Encode the %/day curve as the cost/income model; buyers prefer longer/ITM, sellers harvest the 30→21 zone |
| **Sub-30-DTE ATM is the "red zone" — worst decay, position turns binary** | WORKS | HIGH | yes/research | COIN: 28 DTE, theta ~$97/day rising to ~$160/day @21DTE, −10.5% | Near-the-money holds the most extrinsic value → most theta to lose | Make DTE a first-class gate; never let a long ATM sit into the red zone hoping for recovery |
| **Manage premium-selling positions by ~21 DTE** | WORKS | HIGH | research | 21→0 DTE is the steepest decay + rising gamma | Tastytrade 21-DTE management rule; a forced decision checkpoint | Hard 21-DTE rule for short premium (take profit / close / roll); pair with a ~50%-of-max-profit take |
| **Measure extension in ATR multiples and don't chase** | WORKS | MED | yes/research | AAPL chase drifted ~13.5% of premium day one; ~7–10× daily ATR above the 50-day = heavily extended | Chasing after a run needs the move to keep going right when a pullback is statistically likely | ATR-normalized extension guard before any directional entry; skip if >~1.25× ATR off reference |

### INFRASTRUCTURE & OPS (one-off code bugs — fixed, low strategic carryover)

| Title | Verdict | Conf | Real? | Data | Evidence (1 line) | Next-build applicability |
|---|---|---|---|---|---|---|
| **Scheduler timezone bug ran the whole trading day 1h late** | WORKS(-fixed) | HIGH | yes | daily_reset 09:30 −05:00 (=10:30 ET) before, 09:30 −04:00 (correct) after | CronTrigger baked in machine-local tz; the scheduler tz didn't override per-trigger | Pass explicit market tz to EVERY trigger; verify actual UTC offset of the first fired job |
| **Double-buy/double-close races from two schedulers firing the same second** | WORKS(-fixed) | HIGH | yes | NVDA double-BUY 05-22 (real BUY back into position); rejection 40310000 | Two APScheduler threads both passed "still open?" and acted | Per-contract lock + re-check open status under the lock; treat already-closed rejections as success |
| **PDT + intrinsic immediate-fire caused a same-day-close error cascade** | WORKS(-fixed) | MED | yes | Intrinsic check fired ~2 min on day-1 OTM calls; PDT 40310100; 200+ error lines/day | dte_intrinsic had no days_held floor; day-1 OTM has $0 intrinsic by definition | Gate intrinsic/ITM checks behind days_held≥5; handle PDT explicitly; run ≥$25k to avoid PDT |
| **Hangs/NaN crashes from slow/missing data feeds** | WORKS(-fixed) | MED | no | Finnhub 10s+ → 5s Alpaca; 3 Alpaca calls wrapped at 5s; NaN guard in contract finder; Gemini layer removed | Robustness bugs, not market lessons | Hard timeout + fallback on every external data/LLM call; guard all numeric fields vs NaN; drop low-value LLM layers |

---

## 3. TOP PROVEN TRUTHS (the next bot MUST encode)

Highest-confidence, observed-in-our-system lessons:

1. **Never market-order multi-leg spreads.** Market MLEG combos surrendered 25–54% of mid credit (06-17, −$508 day). Use net-LIMIT near mid; CANCEL+SKIP over a bad fill. *(HIGH, yes)*
2. **Never trust the engine's reported fill — re-validate against the limit floor.** Fills 50–66% below mid were booked while the "skip" log fired **0 times** in 11 logs. Assert booked credit ≥ limit×contracts; reject phantom fills. *(HIGH, yes)*
3. **Derive every stop and all realized P&L from the ACTUAL fill, never the quote.** NVDA booked −$120 on a mark vs ~−$10 real. *(HIGH, yes)*
4. **Judge stops on the underlying stock, with a warmup + 2-cycle confirmation — never on an option mark.** Opening marks produced phantom −14.5% halts, $215-vs-$105 stop trips, and instant −100%/−233% stop-outs. *(HIGH, yes)*
5. **Approve risk on the realized credit, not the quoted credit.** META approved at $117, filled $40 one second later; the 2×-credit stop was already at the line on open. *(HIGH, yes)*
6. **Count BETS, not legs, in every cap/de-dup gate.** A 2-leg spread counting as 2 positions halved capacity and blocked valid entries. *(HIGH, yes)*
7. **Do not build a long-only buy-ATM-momentum-calls engine.** Long calls lost across two independent builds (run2 0W/4L −$3,155; run1 3W/6L −$1,524); buyers structurally win ~25–40%. *(HIGH, yes + research)*
8. **Cap net directional exposure and respect the trend.** A 2:1 short-call skew into a rally produced a 79% loss rate; an all-bullish EOD book is one market-up bet with no correlation cap. *(HIGH, yes)*
9. **Enforce a hard per-position dollar cap at order time — reject, never floor to 1 contract.** One 27.6%-of-equity TSLA fossil drove run-1 to −24.7%. *(HIGH, yes)*
10. **Make payoff geometry positive before launch.** As built, ~$65 capped wins vs ~$121 losses needs a ~65% win-rate to break even — actual was 21%. *(HIGH, yes)*
11. **Compute IV-rank once/day from daily-close HV; sanity-bound and smooth IV/HV before it can flip direction.** Live-mid IV-rank swung 7%→58% in 90 min; IV/HV jumped 0.13→1.67 in 15 min and greenlit a bad entry. *(HIGH, yes)*
12. **Treat DTE/theta as a first-class gate** (research-backed, strategy-neutral): ~6.7%/day @30DTE doubling to ~11%/day @21DTE; manage short premium by 21 DTE; avoid the sub-30 ATM red zone for longs. *(HIGH, research + COIN)*

---

## 4. FAILURE MODES TO NEVER REPEAT (with $ cost)

| Failure mode | Cost | One-line cause |
|---|---|---|
| Market MLEG orders surrendering 25–54% of credit | **−$508** (06-17 day) | Crossing both legs' bid-ask on thin spreads |
| Phantom thin-credit fills → instant credit_stop loop (8 of 19 run-3 trades stopped in 2.6–7.3 min) | **−$1,028** (= 66% of run-3's −$1,554) | Mismarked MLEG entry credit / worst-case ask exits + 2×-credit stop |
| Long-only buy-ATM/near-money calls (run 2) | **−$3,155** (BUY_CALL 0W/4L) | Theta + 50% stop + buyer-side disadvantage |
| Long calls (run 1) | **−$1,524** (BUY_CALL 3W/6L) | Same — confirmed across a second build |
| One oversized TSLA call at 27.6% of equity (run 1) | **≈ −$1,100**, drove run-1 to **−24.7%** | A `max(1, contracts)` floor bypassed the dollar cap |
| ARKK fast sub-day breakdown, no day-1 stop | **−$1,235** (−51% in <1 day) | Flat-% stop only; time-gated exits blind on day 1 |
| Selling call spreads into a rally (run 3 call side) | **−$1,226** (1W/11L) | Fading an uptrend; phantom fills compounded |
| Phantom opening-mark stop booking a fake loss (NVDA) | Booked **−$120** vs real ~−$10 (account actually **+$27.69**) | Stop fired on a garbage $215 opening mark |
| Stranded naked long leg from non-atomic close (NVDA) | ~4h naked exposure, "uncovered" rejection | Closed short leg first in two separate orders |

---

## 5. OPEN / UNRESOLVED (suspected, NOT proven — kept honest and separable)

1. **Is the credit-spread STRUCTURE itself bad, or did only the FILLS fail?** *The single most important open question.* Run-3 lost −$1,554, but **66% of losses came from phantom/instant fills**, and the put-spread side was near breakeven (−$328, 3W/4L). The structure has **not** been shown to fail on clean fills. **Do not conclude credit spreads fail** — re-judge only on trades that survive past ~15 min with honest fills.
2. **Whether the limit-order / atomic-close / stop-warmup / bet-count fixes actually work live.** All are replay/unit-test validated **only**; the bot was stopped 06-14 with ~1 confirmed live close. `UNTESTED`. Needs ~20–30 closed bets to confirm honest fills before any conclusion.
3. **Whether the sell-side redesign actually harvests the VRP.** The pivot from buying to selling premium is theoretically sound (VRP, win-rate asymmetry) but **unproven in our own results** — every run-3 sell-side stat is contaminated by the fill bug. Edge remains **UNPROVEN**.
4. **Achievable credit-to-width in rich IV.** At 0.20 delta the c/w was only 11–16% in the observed low-IV tape; the IVR≥50 / IV/HV≥1.2 / credit≥33% gate produced 0 trades day-1 then was loosened repeatedly. Whether a ≥33% credit/width is reachable in genuinely rich IV was never confirmed.
5. **Anecdotal / low-sample flags (do not over-weight):** Run-2 verdict rests on **5 trades**; run-3 on **19 closed** (1W/10L since 06-18); run-1 BULL_SPREAD "edge" on **4 trades, 3 of which were phantom sub-2-min fills**. The buy-side loss numbers are directionally strong and cross-confirmed by research, but per-bucket magnitudes are small-sample. Judge any next-build strategy only after ~20–30 honestly-filled closed bets, on bottom-line equity — not on marks.
6. **Same-day re-entry cooldown on every close** — logic fix applied, but never observed preventing a live re-entry. `WORKS`-by-replay, unconfirmed live.

---

## 6. The one meta-lesson

Across three runs, the recurring root cause was **not the choice of strategy — it was measurement and execution integrity.** Phantom marks set false stops; phantom fills set false credits; mark-based P&L hid the real bottom line; small samples got over-read. **Before the next bot trades a single contract, the data path must be trustworthy:** real fills read back and asserted, stops and P&L derived only from confirmed fills, marks debounced, and edge judged on bottom-line equity over ≥20–30 honestly-filled closed bets. Get that right first, and the strategy question becomes answerable. Get it wrong, and every strategy looks broken.
