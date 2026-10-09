# Trailing-Stop Choice — Trend-Buyer v1

*Decided 2026-06-27 from comparative research (Chandelier, ATR-trail, Supertrend, Parabolic SAR,
MA-close, Donchian). Companion to [NEXT_BOT_DESIGN.md](NEXT_BOT_DESIGN.md) §0 decision #3.*

Plain English: a "trailing stop" is an exit line that follows the trade up and only ratchets one way, so
you stay in while the trend runs but get out once it turns. "ATR" = Average True Range, the stock's typical
daily price swing — using it makes the exit line auto-widen for jumpy stocks and tighten for calm ones.

---

## THE PICK: Chandelier Exit on the underlying — ATR(22) × 3.0

**Line = (highest daily CLOSE of the underlying since entry) − 3.0 × ATR(22)**, judged on the underlying's
daily close, ratcheting up only.

Why it's the best fit for us:
- We hold **deep-ITM options (delta 0.70–0.80)**, which move almost 1:1 with the stock, so a stock-based
  trail maps cleanly onto the position and dodges the option-mark noise our ledger already bans.
- Our reason to trail is to **catch the rare monster** (the BAC +254% type). Chandelier is purpose-built
  for that: it hangs a fixed 3-ATR cushion below the *running peak*, absorbing normal pullbacks and keeping
  us in the longest while still ratcheting in profit.
- It **won the only direct head-to-head** in the research (Volatility Box, 200 S&P trades 2020–24) on both
  expectancy (+$0.81/trade) and drawdown (−10.2%), and was shaken out least when volatility spiked (16%
  premature stop-outs vs 21–24% for rivals).
- It **pairs with our partial-scaling**: after the +75–100% partial is banked, the runner is "house money,"
  so the wide 3-ATR give-back at the top is an acceptable price for never capping a home run.
- The 22-day period (~one trading month) sits well inside our 60–120 DTE window.

Our entry filters (9/21 EMA + ADX>25 + 200-day regime) already keep us out of the chop where Chandelier is
weak — our entry filter *is* the trend filter the research says it needs. Keep multiplier 3.0; optionally
widen to 3.5–4.0 per-name for high-vol tech if A/B data shows premature stop-outs.

## RUNNER-UP (A/B arm B): ATR-from-the-high — ATR(22) × 2.5

Same engine, tighter cushion, anchored to the highest price. Banks more profit / gives back less at the
top, at the cost of getting shaken out of more would-be monsters. Better if the 3.0× Chandelier consistently
gives back too much at the top. (Parabolic SAR and Supertrend were rejected — both tighten/whipsaw exactly
as a trend matures, fatal to capturing the monster.)

---

## How it composes with the locked design

- **Phase 1 — entry → partial:** position protected ONLY by the initial hard stop (underlying daily close).
  Trail is dormant. (Hard stop = loss-cutter; trail = winner-rider; never both on the full position.)
- **Phase 2 — partial arms the trail:** when the scale-out fires (sell 33–50% at +75–100%), on the same bar:
  bank the partial, **arm the Chandelier on the runner**, retire the original hard stop.
- **Phase 3 — trail the runner:** each daily close, recompute off the **highest close since entry** (keep the
  cushion the trend already earned). Runner rides until the underlying closes below the line, confirmed.
- **Hand-off floor:** at arming, `trail = max(chandelier_line, breakeven_on_underlying)` — the runner can
  never round-trip back to a loss after a partial is taken.
- **Ledger guardrails:** warmup until ATR(22) has a full 22-bar valid history; require **2 consecutive
  daily closes** below the line before exiting (defeats single-bar false-fires).

## Pseudocode (daily-close evaluation of the runner)

```
ATR_PERIOD=22 ; MULTIPLIER=3.0   # 2.5 for A/B arm B
CONFIRM_READS=2 ; WARMUP_BARS=22

def on_daily_close(pos, u):
    closes,highs,lows = u.daily_closes, u.daily_highs, u.daily_lows
    if not pos.trail_armed:
        if partial_just_filled(pos):              # +75-100% target hit, 33-50% sold
            pos.trail_armed=True
            pos.highest_close=max(closes[entry_index:])
            pos.breaks_in_a_row=0
            retire_hard_stop(pos)
        return                                     # still on the hard stop until armed
    if valid_bar_count(u) < WARMUP_BARS: return    # need full ATR history
    pos.highest_close=max(pos.highest_close, closes[-1])
    atr=wilder_atr(highs,lows,closes,ATR_PERIOD)
    line=max(pos.highest_close - MULTIPLIER*atr, breakeven_close(pos))
    pos.breaks_in_a_row = pos.breaks_in_a_row+1 if closes[-1] < line else 0
    if pos.breaks_in_a_row >= CONFIRM_READS: exit_runner(pos)
```
ATR uses Wilder's smoothing. Inputs are monotonic so the line only ratchets up; optionally clamp
`line = max(line, prev_line)` belt-and-suspenders.

## A/B plan (first 20–30 closed runners)

- **Arm A:** Chandelier ATR(22)×3.0. **Arm B:** ATR-from-high ATR(22)×2.5. Everything else identical (only
  trail width/anchor differs). Alternate A/B/A/B (or split by ticker hash) so both see similar regimes.
- **Primary metric — capture ratio:** `captured ÷ peak open profit` per trade (did we ride the monster or
  give it back). **Secondary:** total realized equity; premature stop-outs (exited, then new high within
  ~10 days); avg give-back from peak.
- **Decision:** keep the higher total-equity arm — UNLESS the gap is driven by one or two monster trades, in
  which case favor the arm that caught the monster (our goal is the rare tail). Re-judge at 50+ runners.

## Honest caveats

1. **Small sample, fat tails.** One monster swamps 20–30 trades' averages; don't over-trust early A/B numbers.
2. **Evidence is on stocks, not deep-ITM options.** Deep-ITM tracks the stock closely so it should transfer,
   but leverage + expiration add wrinkles the backtests didn't have.
3. **Expiration can override the trail.** A DTE-based "roll or close before ~21–30 DTE" rule must sit on top.
4. **Conflicting evidence on single stocks.** One 11,000-stock study found a plain 15–25% percentage trail
   beat ATR/Chandelier on single names (different entries, so not directly comparable) — a legit future 3rd arm.
5. **2-confirm cuts both ways:** fewer false-fires, but one extra down-day given back on every real exit. Drop
   to 1 confirm if exits feel late.
6. **"Highest close" vs "highest high":** we anchor to close (ledger rule = no intraday noise), slightly
   tighter than textbook Chandelier — intentional, but not the literal default.
7. **Multiplier is the whole ballgame, tuned on others' data.** 3.0 is a sensible hypothesis, not a settled
   answer — the A/B is how we earn our own number on 100+ of our own trades.
