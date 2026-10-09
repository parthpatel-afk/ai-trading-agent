"""
The day-by-day backtest simulator.

Replays the strategy over history, reusing the LIVE bot's real brain:
  ENTRY  — compute_ema_state → strategy.decide → filters.evaluate → caps → target_delta_for →
           synth.select_contract → estimate_stop_loss_per_contract → size_position → hard_stop_level
  EXIT   — exits.decide_exit (the same pure function the live 15:50 monitor calls)

Convention (documented, no lookahead): each trading day `d`, we use bars *through d's close* as
the decision bar; exits are evaluated first (a held position is managed on the close), then
entries; both fill at day d's close, priced by the synthetic pricer. This compresses the live
"signal on prior close → fill next morning" into one step — a small, symmetric simplification;
a next-open execution lag can be added later. Fills include a slippage haircut off the modeled
(≈ mid) price. Any position still open at the end is closed on the last day (`end_of_backtest`).

Returns a BacktestResult: the trade ledger, the equity curve, a summary, and (optionally) a
per-day entry-decision log used by the trust test.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd

from agent.config import config
from agent.signals.ema import compute_ema_state
from agent.signals import indicators as ind
from agent.signals import filters as filt
from agent import strategy
from agent import exits
from backtest import universe
from backtest.data import store
from backtest.pricing import synth
from backtest.engine.portfolio import Portfolio, SimPosition

# How many trailing bars to feed the indicators. All lookbacks (200-SMA + 20 slope, 252 vol-rank,
# 2*ADX) fit well under this, and EMAs are fully converged — so decisions are identical to using
# full history, but each day is cheap.
_WINDOW = 400


@dataclass
class BacktestResult:
    ledger: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict
    decisions: pd.DataFrame

    def __repr__(self):
        s = self.summary
        return (f"<BacktestResult {s.get('n_trades',0)} trades | "
                f"P&L ${s.get('total_pnl',0):+,.0f} ({s.get('return_pct',0):+.1%}) | "
                f"win {s.get('win_rate',0):.0%} | maxDD {s.get('max_drawdown_pct',0):.1%}>")


# ── helpers ──────────────────────────────────────────────────────────

def _bar_asof(bars: pd.DataFrame, d) -> Optional[pd.Series]:
    """The most recent bar at/at-or-before day d (None if the symbol hasn't listed yet)."""
    sub = bars.loc[:d]
    return sub.iloc[-1] if len(sub) else None


def _slice(bars: pd.DataFrame, d) -> pd.DataFrame:
    return bars.loc[:d].tail(_WINDOW)


def _calendar(bars: Dict[str, pd.DataFrame], start, end) -> List[pd.Timestamp]:
    """Master trading calendar = SPY's sessions (fallback: union of all), within [start, end]."""
    ref = bars.get("SPY")
    idx = ref.index if ref is not None else pd.DatetimeIndex(
        sorted({t for b in bars.values() for t in b.index}))
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    return [d for d in idx if start <= d <= end]


def _sector_of(symbol: str, cfg) -> str:
    # Prefer the live config's sector map; fall back to the universe's richer one.
    return cfg.account.sectors.get(symbol) or universe.sector_of(symbol)


def _caps(port: Portfolio, cfg):
    calls = sum(1 for p in port.positions.values() if p.option_type == "CALL")
    puts = sum(1 for p in port.positions.values() if p.option_type == "PUT")
    by_sector: Dict[str, int] = {}
    for p in port.positions.values():
        sec = _sector_of(p.symbol, cfg)
        by_sector[sec] = by_sector.get(sec, 0) + 1
    return calls, puts, by_sector


def _direction_ok(option_type, calls, puts, cfg) -> bool:
    allowed = max(1, round(cfg.account.max_positions * cfg.account.direction_cap_pct))
    same = (calls if option_type == "CALL" else puts) + 1
    return same <= allowed


def _price_map(port: Portfolio, bars, d) -> Dict[str, float]:
    pm = {}
    for sym, p in port.positions.items():
        row = _bar_asof(bars[sym], d)
        spot = float(row["close"]) if row is not None else p.entry_underlying
        pm[sym] = p.mark(spot, d.date() if hasattr(d, "date") else d)
    return pm


# ── EXITS (mirror run_close_monitor) ─────────────────────────────────

def _run_exits(port: Portfolio, bars, d, cfg, slippage):
    dd = d.date() if hasattr(d, "date") else d
    for sym in list(port.positions.keys()):
        pos = port.positions[sym]
        b = bars[sym]
        sl = _slice(b, d)
        if sl.empty:
            continue
        close = float(sl["close"].iloc[-1])
        chand_atr = ind.atr(sl, cfg.exits.chandelier_atr_period) or pos.atr_at_entry or 0.0
        ema21 = float(sl["close"].astype(float).ewm(span=cfg.strategy.ema_slow, adjust=False).mean().iloc[-1])
        opt_px = synth.reprice(pos.contract, close, dd)
        gain = (opt_px - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
        dte = max(0, (pos.contract.expiration - dd).days)

        dec = exits.decide_exit(pos.state(), close, chand_atr, ema21, gain, dte, cfg)

        if dec.action in (exits.TIME_EXIT, exits.HARD_STOP, exits.TRAIL_EXIT):
            port.close_full(pos, opt_px * (1 - slippage), dd, close, dec.reason)
        elif dec.action == exits.PARTIAL_ARM:
            _take_partial_and_arm(port, pos, opt_px * (1 - slippage), dec.new_peak, cfg)
        else:  # WAIT_* / HOLD — persist streaks + peak
            pos.peak_close = dec.new_peak
            pos.stop_streak = dec.new_stop_streak
            pos.trail_streak = dec.new_trail_streak


def _take_partial_and_arm(port, pos, opt_px, peak, cfg):
    """Mirror scheduler._take_partial_and_arm: sell a fraction if we hold >=2, else just arm."""
    n0 = pos.contracts_initial
    partial_qty = int(n0 * cfg.exits.partial_scale_fraction)
    if partial_qty >= 1 and pos.contracts - partial_qty >= 1:
        port.take_partial(pos, partial_qty, opt_px)
    pos.partial_done = True
    pos.trail_armed = True
    pos.peak_close = peak
    pos.trail_streak = 0


# ── ENTRIES (mirror run_open_routine) ────────────────────────────────

def _run_entries(port: Portfolio, bars, d, cfg, slippage, equity, decisions: Optional[List]):
    dd = d.date() if hasattr(d, "date") else d
    calls, puts, by_sector = _caps(port, cfg)

    for symbol in cfg.strategy.watchlist:
        b = bars.get(symbol)
        if b is None:
            continue
        sl = _slice(b, d)
        ema = compute_ema_state(symbol, sl)
        if ema is None:
            continue
        held = port.position_for(symbol)
        decision = strategy.decide(ema, held.option_type if held else None)
        action, reason, adx = decision.action, decision.reason, None

        if decision.action in ("BUY_CALL", "BUY_PUT"):
            direction = "bull" if decision.option_type == "CALL" else "bear"
            fr = filt.evaluate(sl, direction, cfg)
            adx = fr.metrics.get("adx")
            if not fr.passed:
                action, reason = "SKIP", f"filtered: {fr.reason}"
            else:
                action, reason = _try_enter(port, symbol, decision, fr, sl, dd, cfg, slippage,
                                            equity, calls, puts, by_sector)
                if action.startswith("BUY"):
                    # refresh running caps so later symbols see this position
                    calls, puts, by_sector = _caps(port, cfg)

        if decisions is not None and (ema.fresh or action.startswith("BUY")):
            decisions.append({
                "date": dd, "symbol": symbol, "close": ema.close,
                "ema9": ema.ema_fast, "ema21": ema.ema_slow, "cross": ema.cross,
                "age": ema.cross_age_days, "fresh": ema.fresh, "adx": adx,
                "action": action, "reason": reason,
            })


def _try_enter(port, symbol, decision, fr, sl, dd, cfg, slippage, equity,
               calls, puts, by_sector):
    """Caps → contract → sizing → fill. Returns (action, reason)."""
    otype = decision.option_type
    held = port.position_for(symbol)

    # portfolio caps (only gate genuinely NEW symbols; a flip reuses the slot)
    if not held:
        if len(port.positions) >= cfg.account.max_positions:
            return "SKIP", f"max positions {cfg.account.max_positions}"
        if not _direction_ok(otype, calls, puts, cfg):
            return "SKIP", f"direction cap ({otype})"
        sec = _sector_of(symbol, cfg)
        if by_sector.get(sec, 0) >= cfg.account.max_per_sector:
            return "SKIP", f"sector cap ({sec})"

    close = float(sl["close"].iloc[-1])
    sigma = synth.hv_sigma(sl["close"], cfg=cfg)
    if not sigma:
        return "SKIP", "no HV"

    # flip: close the opposite side first (at today's close)
    if held:
        opt_px = synth.reprice(held.contract, close, dd)
        port.close_full(held, opt_px * (1 - slippage), dd, close, "ema_flip")

    tdelta = strategy.target_delta_for(fr.metrics.get("vol_rank"), cfg)
    contract = synth.select_contract(symbol, otype, close, sigma, dd, tdelta, cfg)
    if contract is None:
        return "SKIP", "no contract"

    atr_val = fr.metrics.get("atr") or ind.atr(sl, cfg.filters.atr_period) or 0.0
    delta = abs(contract.entry_delta)
    fill_px = contract.entry_price * (1 + slippage)           # per share, worse than mid
    ask_per_contract = fill_px * 100
    est_loss = strategy.estimate_stop_loss_per_contract(delta, atr_val, cfg.exits.hard_stop_atr_mult)
    sized = strategy.size_position(equity, port.cash, ask_per_contract, est_loss, cfg)
    if sized.contracts < 1:
        return "SKIP", f"sizing: {sized.reason}"

    hard_stop = strategy.hard_stop_level(close, atr_val, otype, cfg)
    pos = SimPosition(
        symbol=symbol, contract=contract, option_type=otype,
        contracts=sized.contracts, contracts_initial=sized.contracts,
        entry_price=fill_px, premium_paid=fill_px * 100 * sized.contracts,
        entry_date=dd, entry_underlying=close, atr_at_entry=atr_val,
        hard_stop_price=hard_stop, delta_at_entry=round(delta, 3),
        hv_at_entry=sigma, vol_rank_at_entry=fr.metrics.get("vol_rank"),
        adx_at_entry=fr.metrics.get("adx"), peak_close=close,
    )
    port.open(pos)
    return decision.action, decision.reason


# ── driver ───────────────────────────────────────────────────────────

def run_backtest(start="2020-01-01", end="2026-12-31", symbols: Optional[List[str]] = None,
                 cfg=None, slippage_pct: float = 0.01, initial_capital: Optional[float] = None,
                 record_decisions: bool = False) -> BacktestResult:
    cfg = cfg or config
    watch = symbols or list(cfg.strategy.watchlist)
    # let a custom symbol list drive the run without editing the global watchlist
    if symbols is not None:
        cfg = cfg.clone_with({"strategy.watchlist": list(symbols)})
    initial_capital = float(initial_capital if initial_capital is not None else cfg.account.initial_capital)

    bars = {s: store.load_bars(s) for s in set(watch) | {"SPY"}}
    bars = {s: b for s, b in bars.items() if b is not None}
    cal = _calendar(bars, start, end)
    if not cal:
        raise ValueError(f"no trading days in [{start}, {end}] — is the cache populated?")

    port = Portfolio(initial_capital)
    equity_rows, decision_rows = [], ([] if record_decisions else None)

    for d in cal:
        _run_exits(port, bars, d, cfg, slippage_pct)
        equity = port.equity(_price_map(port, bars, d))     # equity for sizing (post-exit)
        _run_entries(port, bars, d, cfg, slippage_pct, equity, decision_rows)
        pm = _price_map(port, bars, d)
        equity_rows.append({"date": d, "equity": round(port.equity(pm), 2),
                            "cash": round(port.cash, 2), "positions": len(port.positions)})

    # close anything still open on the last day
    last = cal[-1]; dd = last.date()
    for sym in list(port.positions.keys()):
        pos = port.positions[sym]
        row = _bar_asof(bars[sym], last)
        spot = float(row["close"]) if row is not None else pos.entry_underlying
        port.close_full(pos, synth.reprice(pos.contract, spot, dd) * (1 - slippage_pct),
                        dd, spot, "end_of_backtest")

    ledger = pd.DataFrame(port.ledger)
    equity_curve = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    summary = _summarize(ledger, equity_curve, initial_capital, start, end)
    decisions = pd.DataFrame(decision_rows) if decision_rows else pd.DataFrame()
    return BacktestResult(ledger, equity_curve, summary, decisions)


def _summarize(ledger: pd.DataFrame, equity_curve: pd.DataFrame,
               initial_capital: float, start, end) -> dict:
    s = {"start": str(start), "end": str(end), "initial_capital": initial_capital,
         "n_trades": int(len(ledger))}
    if ledger.empty:
        s.update({"total_pnl": 0.0, "return_pct": 0.0, "win_rate": 0.0,
                  "final_equity": initial_capital, "max_drawdown_pct": 0.0})
        return s
    wins = ledger[ledger["pnl"] > 0]; losses = ledger[ledger["pnl"] <= 0]
    total = float(ledger["pnl"].sum())
    s["total_pnl"] = round(total, 2)
    s["final_equity"] = round(initial_capital + total, 2)
    s["return_pct"] = round(total / initial_capital, 4)
    s["win_rate"] = round(len(wins) / len(ledger), 4)
    s["avg_win"] = round(float(wins["pnl"].mean()), 2) if len(wins) else 0.0
    s["avg_loss"] = round(float(losses["pnl"].mean()), 2) if len(losses) else 0.0
    s["avg_pnl"] = round(float(ledger["pnl"].mean()), 2)
    gp = float(wins["pnl"].sum()); gl = abs(float(losses["pnl"].sum()))
    s["profit_factor"] = round(gp / gl, 2) if gl else float("inf")
    s["calls"] = int((ledger["direction"] == "CALL").sum())
    s["puts"] = int((ledger["direction"] == "PUT").sum())
    s["avg_hold_days"] = round(float(ledger["hold_days"].mean()), 1)
    s["exit_reasons"] = ledger["exit_reason"].value_counts().to_dict()
    if not equity_curve.empty:
        eq = equity_curve["equity"]
        s["max_drawdown_pct"] = round(float(((eq - eq.cummax()) / eq.cummax()).min()), 4)
    return s
