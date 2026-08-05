# ==============================================================================
# intrabar_risk.py
# ==============================================================================
# Phase 2, Item 14 -- pessimistic intrabar fills / honest mark-to-market.
#
# WHAT THE COMPLIANCE CHECK CURRENTLY MISSES
# ------------------------------------------
# ftmo_compliance builds its equity curve from trade events plus injected
# midnight checkpoints. Two consequences, both understating risk:
#
#   1. NO INTRABAR EXCURSION. The word 'high' does not appear in the module.
#      Equity is marked on close-like prices only, so a position that goes
#      deeply underwater within a bar and recovers by the close registers no
#      drawdown at all. A real broker marks to market continuously, and a 5%
#      breach at ANY moment ends the challenge -- there is no credit for
#      recovering before the candle finished.
#
#   2. STALE CHECKPOINT PRICES. At a checkpoint there is no entry or exit
#      price, so the code falls back to `last_price` -- the price from the last
#      TRADE event, which may be days old:
#
#          event_price = event.get('exit_price') or event.get('entry_price') or last_price
#
#      A position held across a week is therefore marked at the price it was
#      opened at, every single day, until it closes. Its entire adverse
#      excursion is invisible to the daily-loss rule.
#
# validate() does not even accept price data, so it structurally cannot do
# better. That is the gap this module fills.
#
# WHAT COUNTS AS PESSIMISTIC AND WHAT IS SIMPLY TRUE
# --------------------------------------------------
# For a SINGLE open position, marking at the bar's adverse extreme is not a
# pessimistic assumption -- price genuinely traded there, so the account
# genuinely held that loss at some instant during the bar. Nothing is being
# assumed.
#
# For SEVERAL positions open at once on different instruments, assuming they
# all reach their worst point in the same instant IS conservative; the true
# simultaneous low is somewhere between the close-only and adverse-extreme
# paths. Both bounds are reported, and which one to plan against is a judgement
# about how much margin you want, not something this module should decide.
#
# MAE
# ---
# Maximum adverse excursion per trade -- how far against you a position went
# before it closed. A trade booked as a winner that first ran 80 pips against
# you would have been a loser under any stop tighter than that, and it consumed
# daily-loss budget on the way. Backtests that only record entry and exit hide
# this completely.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MODE_CLOSE = 'close'
MODE_ADVERSE = 'adverse'
VALID_MODES = (MODE_CLOSE, MODE_ADVERSE)


@dataclass
class TradeExcursion:
    entry_date: Any
    exit_date: Any
    symbol: str
    size: float
    entry_price: float
    exit_price: float
    realised_pnl: float
    mae: float                  # worst unrealised loss, in currency (<= 0)
    mfe: float                  # best unrealised gain, in currency (>= 0)
    mae_price: float            # price at the worst point
    bars: int

    @property
    def hidden_loss(self) -> float:
        """
        How much adverse excursion the entry/exit record conceals.

        Zero for a trade that never traded against you. Large on a winner that
        first went deeply underwater -- exactly the trade a close-only equity
        curve reports as riskless.
        """
        return max(0.0, -self.mae - max(0.0, -self.realised_pnl))


@dataclass
class IntrabarReport:
    account_size: float
    trades_analysed: int = 0
    trades_unmatched: int = 0
    close_only_max_daily_loss_pct: float = 0.0
    adverse_max_daily_loss_pct: float = 0.0
    close_only_max_drawdown_pct: float = 0.0
    adverse_max_drawdown_pct: float = 0.0
    days_flipped: List[Dict[str, Any]] = field(default_factory=list)
    worst_excursions: List[TradeExcursion] = field(default_factory=list)
    total_hidden_loss: float = 0.0
    error: Optional[str] = None

    @property
    def verdict_changes(self) -> bool:
        return bool(self.days_flipped)

    def summary(self, limit_pct: float = 5.0) -> str:
        L = [f"\n{'=' * 70}", "  INTRABAR RISK ANALYSIS", '=' * 70]
        if self.error:
            L += [f"  [ERROR] {self.error}", '=' * 70]
            return '\n'.join(L)

        L.append(f"  Account: ${self.account_size:,.0f}   "
                 f"Trades: {self.trades_analysed}"
                 + (f"   Unmatched: {self.trades_unmatched}"
                    if self.trades_unmatched else ""))
        L.append("")
        L.append(f"  {'':22} {'close-only':>12} {'adverse':>12}")
        L.append(f"  {'worst daily loss %':22} "
                 f"{self.close_only_max_daily_loss_pct:12.2f} "
                 f"{self.adverse_max_daily_loss_pct:12.2f}")
        L.append(f"  {'max drawdown %':22} "
                 f"{self.close_only_max_drawdown_pct:12.2f} "
                 f"{self.adverse_max_drawdown_pct:12.2f}")

        if self.total_hidden_loss:
            L.append("")
            L.append(f"  Adverse excursion hidden by entry/exit records: "
                     f"${self.total_hidden_loss:,.0f}")

        if self.days_flipped:
            L.append("")
            L.append(f"  [BREACH] {len(self.days_flipped)} day(s) pass on close prices "
                     f"but breach the {limit_pct}% limit intrabar:")
            for d in self.days_flipped[:8]:
                L.append(f"    {d['date']}  close-only {d['close_pct']:5.2f}%  "
                         f"-> intrabar {d['adverse_pct']:5.2f}%")
            L.append("")
            L.append("  These are real breaches. A broker marks to market")
            L.append("  continuously; recovering before the candle closed earns")
            L.append("  no credit.")
        else:
            L.append("")
            L.append("  No day crosses the limit intrabar that did not already")
            L.append("  cross it on close prices.")

        if self.worst_excursions:
            L.append("")
            L.append("  Worst adverse excursions:")
            for e in self.worst_excursions[:5]:
                tag = " (booked as a WINNER)" if e.realised_pnl > 0 else ""
                L.append(f"    {str(e.entry_date)[:16]} {e.symbol:9} "
                         f"MAE ${e.mae:>10,.0f}  realised ${e.realised_pnl:>9,.0f}{tag}")
        L.append('=' * 70)
        return '\n'.join(L)


# ==============================================================================
# CORE
# ==============================================================================

def _f(value, default: float = 0.0) -> float:
    """
    Coerce a pandas cell to a concrete float.

    Values pulled from iterrows() or .get() arrive typed as Any | Series |
    Unknown, and NaN slips through a plain `or default` because NaN is truthy.
    Centralising the conversion handles both, and gives a type checker
    something definite to work with instead of 23 separate complaints.
    """
    try:
        if value is None:
            return default
        out = float(value)
        return default if pd.isna(out) else out
    except (TypeError, ValueError):
        return default


def _prep_trades(trades) -> pd.DataFrame:
    df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades or [])
    if df.empty:
        return df
    for c in ('entry_date', 'exit_date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    if 'symbol' not in df.columns:
        df['symbol'] = 'UNKNOWN'
    return df.sort_values('entry_date').reset_index(drop=True)


def _prep_prices(price_data) -> Dict[str, pd.DataFrame]:
    """Accept a single frame or {symbol: frame}."""
    if isinstance(price_data, pd.DataFrame):
        return {'*': price_data}
    return dict(price_data or {})


def _bars_for(prices: Dict[str, pd.DataFrame], symbol: str) -> Optional[pd.DataFrame]:
    # Explicit None check, not `a or b`: the truth value of a DataFrame is
    # ambiguous and pandas raises rather than falling through.
    bars = prices.get(symbol)
    if bars is None:
        bars = prices.get('*')
    return bars


def trade_excursions(trades, price_data) -> List[TradeExcursion]:
    """
    Per-trade maximum adverse and favourable excursion.

    Requires OHLC covering the holding period. Trades whose bars cannot be
    found are skipped rather than assigned a zero excursion -- pretending an
    unmeasured trade had no adverse move is the failure mode this exists to
    remove.
    """
    df = _prep_trades(trades)
    prices = _prep_prices(price_data)
    out: List[TradeExcursion] = []
    if df.empty:
        return out

    for _, t in df.iterrows():
        bars = _bars_for(prices, str(t.get('symbol') or 'UNKNOWN'))
        if bars is None or bars.empty:
            continue
        if not {'high', 'low'} <= set(bars.columns):
            continue

        e, x = t.get('entry_date'), t.get('exit_date')
        # bool() is explicit: pd.isna can return an array for some inputs,
        # and a bare `or` on that is ambiguous.
        if bool(pd.isna(e)) or bool(pd.isna(x)):
            continue

        window = bars[(bars.index >= e) & (bars.index <= x)]
        if window.empty:
            continue

        size = _f(t.get('size'))
        entry_price = _f(t.get('entry_price'))
        if size == 0 or entry_price == 0:
            continue

        # Long: the adverse extreme is the window low. Short: the high.
        if size > 0:
            worst_price = _f(window['low'].min())
            best_price = _f(window['high'].max())
        else:
            worst_price = _f(window['high'].max())
            best_price = _f(window['low'].min())

        mae = (worst_price - entry_price) * size
        mfe = (best_price - entry_price) * size
        realised = (_f(t.get('pnl')) if t.get('pnl') is not None
                    else (_f(t.get('exit_price'), entry_price) - entry_price) * size)

        out.append(TradeExcursion(
            entry_date=e, exit_date=x, symbol=str(t.get('symbol', 'UNKNOWN')),
            size=size, entry_price=entry_price,
            exit_price=_f(t.get('exit_price'), entry_price),
            realised_pnl=realised, mae=min(0.0, mae), mfe=max(0.0, mfe),
            mae_price=worst_price, bars=len(window),
        ))
    return out


def equity_path(trades, price_data, initial_balance: float,
                mode: str = MODE_ADVERSE) -> pd.DataFrame:
    """
    Bar-by-bar equity, marking open positions at every bar.

    mode='close'    mark at each bar's close -- approximates what the existing
                    compliance curve does, minus its stale-price problem.
    mode='adverse'  mark at each bar's adverse extreme. For one open position
                    this is not an assumption: price traded there.

    Returns a frame indexed by timestamp with balance, equity and open_count.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    df = _prep_trades(trades)
    prices = _prep_prices(price_data)
    if df.empty:
        return pd.DataFrame()

    # Union of all bar timestamps spanned by any trade.
    spans = []
    for sym in df['symbol'].unique():
        bars = _bars_for(prices, sym)
        if bars is None or bars.empty:
            continue
        lo, hi = df['entry_date'].min(), df['exit_date'].max()
        spans.append(bars[(bars.index >= lo) & (bars.index <= hi)].index)
    if not spans:
        return pd.DataFrame()

    timeline = spans[0]
    for s in spans[1:]:
        timeline = timeline.union(s)
    timeline = timeline.sort_values()

    balance = float(initial_balance)
    rows = []
    exits_by_time = df.set_index('exit_date', drop=False)

    for ts in timeline:
        # Realise anything that closed at or before this bar.
        closed = exits_by_time[(exits_by_time['exit_date'] <= ts)]
        realised = _f(closed['pnl'].sum()) if 'pnl' in closed.columns else 0.0
        bal = _f(initial_balance) + realised

        # Mark whatever is still open.
        open_now = df[(df['entry_date'] <= ts) & (df['exit_date'] > ts)]
        unreal = 0.0
        for _, t in open_now.iterrows():
            bars = _bars_for(prices, str(t.get('symbol') or 'UNKNOWN'))
            if bars is None or ts not in bars.index:
                continue
            bar = bars.loc[ts]
            size = _f(t.get('size'))
            entry_price = _f(t.get('entry_price'))
            if mode == MODE_CLOSE:
                mark = _f(bar['close'])
            else:
                mark = _f(bar['low']) if size > 0 else _f(bar['high'])
            unreal += (mark - entry_price) * size

        rows.append({'timestamp': ts, 'balance': bal,
                     'equity': bal + unreal, 'open_count': len(open_now)})

    if not rows:
        # An empty timeline means the price data does not overlap the trades.
        # Returning an unindexed empty frame here would raise on set_index and
        # look like a bug in the caller.
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('timestamp')


def analyze(trades, price_data, account_size: float = 100_000,
            max_daily_loss_pct: float = 5.0,
            initial_balance: Optional[float] = None) -> IntrabarReport:
    """
    Compare the close-only and adverse-extreme views of the same trade history.

    The headline output is days_flipped: days that pass the daily-loss rule on
    close prices and breach it intrabar. Those are real breaches.
    """
    initial_balance = initial_balance if initial_balance is not None else account_size
    rep = IntrabarReport(account_size=account_size)

    df = _prep_trades(trades)
    if df.empty:
        rep.error = "No trades to analyse"
        return rep

    prices = _prep_prices(price_data)
    if not prices:
        rep.error = ("No price data supplied. Intrabar risk cannot be measured "
                     "from entry and exit records alone -- that is the whole gap.")
        return rep

    exc = trade_excursions(df, prices)
    rep.trades_analysed = len(exc)
    rep.trades_unmatched = len(df) - len(exc)
    rep.worst_excursions = sorted(exc, key=lambda e: e.mae)[:10]
    rep.total_hidden_loss = float(sum(e.hidden_loss for e in exc))

    close_path = equity_path(df, prices, initial_balance, MODE_CLOSE)
    adv_path = equity_path(df, prices, initial_balance, MODE_ADVERSE)
    if close_path.empty or adv_path.empty:
        rep.error = "Price data does not overlap the trade dates"
        return rep

    limit_amount = max_daily_loss_pct / 100.0 * initial_balance

    def daily(path):
        g = path.groupby(path.index.date)
        # Anchor on the balance carried into the day, matching the FTMO rule.
        return g.agg(min_equity=('equity', 'min'),
                     open_balance=('balance', 'first'))

    cd, ad = daily(close_path), daily(adv_path)
    joined = cd.join(ad, lsuffix='_close', rsuffix='_adv', how='outer')

    for date, r in joined.iterrows():
        anchor = _f(r.get('open_balance_close'), initial_balance)
        c_loss = (anchor - _f(r['min_equity_close'])) / initial_balance * 100
        a_loss = (anchor - _f(r['min_equity_adv'])) / initial_balance * 100
        rep.close_only_max_daily_loss_pct = max(rep.close_only_max_daily_loss_pct, float(c_loss))
        rep.adverse_max_daily_loss_pct = max(rep.adverse_max_daily_loss_pct, float(a_loss))
        if c_loss <= max_daily_loss_pct < a_loss:
            rep.days_flipped.append({
                'date': str(date), 'close_pct': round(float(c_loss), 2),
                'adverse_pct': round(float(a_loss), 2),
            })

    def max_dd(path):
        eq = path['equity'].to_numpy()
        if len(eq) == 0:
            return 0.0
        peak = np.maximum.accumulate(np.concatenate([[initial_balance], eq]))[1:]
        return float(np.max((peak - eq) / np.maximum(peak, 1e-9)) * 100)

    rep.close_only_max_drawdown_pct = max_dd(close_path)
    rep.adverse_max_drawdown_pct = max_dd(adv_path)
    return rep
