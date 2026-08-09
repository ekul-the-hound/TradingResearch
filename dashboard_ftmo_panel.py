# ==============================================================================
# dashboard_ftmo_panel.py
# ==============================================================================
# Phase 0, Item 3 -- dashboard FTMO proxy fix.
#
# WHAT WAS WRONG
# --------------
# Both dashboards rendered PASS/FAIL badges for FTMO rules without ever calling
# FTMOComplianceChecker, even though it was imported and available.
#
# react_dashboard2.PgFTMOPortfolio:
#     d_ok = dd_pct < 0.05    # "Daily<5%"  -- this is TOTAL max drawdown
#     t_ok = dd_pct < 0.10    # "Total<10%" -- the SAME number, second threshold
#   The daily-loss column never looked at a daily boundary at all. Any strategy
#   with a total drawdown between 5% and 10% displayed Daily=FAIL / Total=PASS
#   purely as an artifact of testing one quantity twice. Min-trading-days was
#   absent from the table entirely, so a 4-rule check displayed 3 rules.
#
# dashboard_react.render_ftmo:
#     daily_ok    = scaled_dd < 5
#     min_days_ok = result.total_trades >= 4      # trade COUNT as trading DAYS
#   Four trades inside one session satisfied a rule requiring four distinct
#   days. This file already built `trades_df = pd.DataFrame(result.trades)`
#   on the line above and then ignored it.
#
# WHY IT WAS LIKE THAT
# --------------------
# The `backtest_results` table stores summary statistics only -- no trade list.
# So a dashboard reading from that DB genuinely cannot run the real checker.
# The proxy was a workaround for a missing data path, not carelessness. This
# module fixes the dashboard half; the persistence gap is noted below.
#
# THE RULE THIS MODULE FOLLOWS
# ----------------------------
# Render a verdict only when a real verdict was computed. When trades are not
# reachable, return UNAVAILABLE with a reason and let the UI say so. A blank
# cell is honest; a green badge derived from an unrelated number is not.
#
# TRADE SOURCES, IN ORDER OF FIDELITY
#   1. live_trades -- an in-process result carrying .trades (dashboard_react).
#                     Exact.
#   2. decay_db    -- strategy_trades in the decay database. Has entry_time,
#                     exit_time, pnl, size, is_long but NO entry_price.
#                     For FX this is exact: fees are $5/lot from size and
#                     spread is lots * pip_value, both price-independent.
#                     For crypto/indices fees are notional-based, so a
#                     reconstructed price makes them approximate -- flagged,
#                     never hidden.
#   3. none        -- UNAVAILABLE.
#
# KNOWN GAP (not fixed here, belongs on the roadmap):
#   backtest_results persists no trade list. Until it does, the main dashboard
#   can only show real compliance for strategies whose trades were separately
#   saved via DecayCalculator.save_trades(). This is the same root cause as the
#   canonical_result.py synthetic-returns fallback: trades not flowing through.
# ==============================================================================

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

ACCOUNT_SIZES = [10_000, 25_000, 50_000, 100_000, 200_000]

SOURCE_LIVE = 'live_trades'
SOURCE_RESULTS_DB = 'results_db'
SOURCE_DECAY_DB = 'decay_db'
SOURCE_NONE = 'none'

# Reconstructed price for DB-sourced trades. Exact for FX (fees ignore price),
# approximate for notional-fee asset classes.
_SYNTHETIC_PRICE = 1.0


@dataclass
class PanelResult:
    """What the FTMO panel should render. Never contains a fabricated verdict."""
    available: bool
    rows: List[dict] = field(default_factory=list)
    source: str = SOURCE_NONE
    reason: str = ''             # why unavailable
    approximate: bool = False    # fees reconstructed, not exact
    approximate_note: str = ''
    strategy_id: str = ''
    phase: str = 'challenge'

    @property
    def any_pass(self) -> bool:
        return any(r['passed'] for r in self.rows)


def unavailable(reason: str, strategy_id: str = '') -> PanelResult:
    return PanelResult(available=False, reason=reason, strategy_id=strategy_id)


# ==============================================================================
# CORE: run the real checker
# ==============================================================================

def rows_from_trades(
    trades,
    phase: str = 'challenge',
    account_sizes: Optional[List[int]] = None,
    strategy_id: str = '',
    source: str = SOURCE_LIVE,
    approximate: bool = False,
    approximate_note: str = '',
) -> PanelResult:
    """
    Run FTMOComplianceChecker across account sizes and return renderable rows.

    Args:
        trades: DataFrame or list-of-dicts with entry_date, exit_date,
                entry_price, size (exit_price or pnl also required).
        phase: 'challenge' (+10%) or 'verification' (+5%).

    Returns:
        PanelResult. available=False if the checker could not run -- the UI
        must render that as "unavailable", not as a failing verdict.
    """
    account_sizes = account_sizes or ACCOUNT_SIZES

    try:
        from ftmo_compliance import FTMOComplianceChecker
    except Exception as e:
        return unavailable(f"ftmo_compliance unavailable: {e}", strategy_id)

    df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades or [])
    if df is None or df.empty:
        return unavailable("No trades recorded for this strategy", strategy_id)

    checker = FTMOComplianceChecker()
    rows = []

    for size in account_sizes:
        try:
            r = checker.validate(df, account_size=size, phase=phase)
        except Exception as e:
            # One size failing should not fabricate results for the others.
            return unavailable(f"Compliance check failed: {e}", strategy_id)

        rows.append({
            'account_size': size,
            'final_equity': r.final_equity,
            'final_return_pct': r.final_return_pct,
            'daily_ok': bool(r.daily_loss_ok),
            'total_ok': bool(r.total_drawdown_ok),
            'min_days_ok': bool(r.min_days_ok),
            'profit_ok': bool(r.profit_target_ok),
            'passed': bool(r.passed),
            'max_daily_loss_pct': r.max_daily_loss_pct,
            'max_total_drawdown_pct': r.max_total_drawdown_pct,
            'trading_days': r.trading_days,
        })

    return PanelResult(
        available=True,
        rows=rows,
        source=source,
        approximate=approximate,
        approximate_note=approximate_note,
        strategy_id=strategy_id,
        phase=phase,
    )


# ==============================================================================
# DB PATH: reconstruct trades from the decay database
# ==============================================================================

def _is_fx(symbol: str) -> bool:
    try:
        from ftmo_compliance import detect_asset_class, AssetClass
        return detect_asset_class(symbol or '') == AssetClass.FX
    except Exception:
        s = (symbol or '').upper().replace('-', '').replace('/', '')
        majors = ('EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD')
        return len(s) == 6 and s[:3] in majors and s[3:] in majors


def rows_from_decay_db(
    db_path: str,
    strategy_id: str,
    symbol: Optional[str] = None,
    phase: str = 'challenge',
    account_sizes: Optional[List[int]] = None,
) -> PanelResult:
    """
    Build a compliance panel from persisted strategy_trades.

    strategy_trades has no entry_price, so prices are reconstructed such that
    (exit_price - entry_price) * size reproduces the recorded pnl exactly.
    Daily-loss and drawdown math is therefore exact. Fees are exact for FX
    (price-independent) and approximate otherwise (notional-based) -- the
    approximate flag is set so the UI can label it.
    """
    if not db_path or not os.path.exists(db_path):
        return unavailable("Decay database not found; no persisted trades", strategy_id)

    sql = """
        SELECT symbol, entry_time, exit_time, pnl, size, is_long
        FROM strategy_trades
        WHERE strategy_id = ?
    """
    params = [strategy_id]
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    sql += " ORDER BY exit_time ASC"

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        raw = [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception as e:
        return unavailable(f"Could not read persisted trades: {e}", strategy_id)
    finally:
        if conn is not None:
            conn.close()

    if not raw:
        return unavailable(
            "No persisted trades for this strategy. "
            "Run DecayCalculator.save_trades() after a backtest to enable "
            "real compliance checking.",
            strategy_id,
        )

    rows = []
    for t in raw:
        size = t.get('size') or 0
        if not size:
            continue
        signed = abs(size) * (1 if t.get('is_long') else -1)
        pnl = t.get('pnl') or 0.0
        entry_price = _SYNTHETIC_PRICE
        # Chosen so (exit - entry) * signed_size == pnl exactly.
        exit_price = entry_price + (pnl / signed)
        rows.append({
            'entry_date': t.get('entry_time') or t.get('exit_time'),
            'exit_date': t.get('exit_time'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': signed,
            'symbol': t.get('symbol') or symbol or 'EUR-USD',
        })

    if not rows:
        return unavailable("Persisted trades have no usable position sizes", strategy_id)

    df = pd.DataFrame(rows)
    non_fx = sorted({s for s in df['symbol'].unique() if not _is_fx(s)})
    approximate = bool(non_fx)
    note = ''
    if approximate:
        note = (f"Fees approximate for {', '.join(non_fx)}: entry prices are not "
                f"persisted and these asset classes charge notional-based fees. "
                f"Drawdown and daily-loss figures are exact.")

    return rows_from_trades(
        df,
        phase=phase,
        account_sizes=account_sizes,
        strategy_id=strategy_id,
        source=SOURCE_DECAY_DB,
        approximate=approximate,
        approximate_note=note,
    )



# ==============================================================================
# RESULTS DB PATH: exact trades, no reconstruction
# ==============================================================================

def rows_from_results_db(
    db_path: str,
    variant_id: Optional[str] = None,
    strategy_name: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    phase: str = 'challenge',
    account_sizes: Optional[List[int]] = None,
) -> PanelResult:
    """
    Build a panel from backtest_trades in the main results database.

    This is the highest-fidelity persisted source. Unlike strategy_trades in
    the decay DB, it stores BOTH entry and exit prices, so fees are exact for
    every asset class -- no reconstruction, no approximate flag, no caveat for
    notional-fee instruments.

    Returns UNAVAILABLE (never a fabricated verdict) when the table is absent
    or the backtest predates trade persistence.
    """
    sid = variant_id or strategy_name or ''
    if not db_path or not os.path.exists(db_path):
        return unavailable("Results database not found", sid)

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row

        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_trades'"
        ).fetchone()
        if not exists:
            return unavailable(
                "No backtest_trades table in the results database. Trade "
                "persistence is applied but this database predates it -- "
                "re-run a backtest to populate it.", sid)

        q = "SELECT id FROM backtest_results WHERE 1=1"
        p: List[Any] = []
        if variant_id:
            q += " AND variant_id = ?"; p.append(variant_id)
        if strategy_name and not variant_id:
            q += " AND strategy_name = ?"; p.append(strategy_name)
        if symbol:
            q += " AND symbol = ?"; p.append(symbol)
        if timeframe:
            q += " AND timeframe = ?"; p.append(timeframe)
        q += " ORDER BY id DESC LIMIT 1"

        head = conn.execute(q, p).fetchone()
        if not head:
            return unavailable("No matching backtest in the results database", sid)

        raw = [dict(r) for r in conn.execute(
            "SELECT * FROM backtest_trades WHERE backtest_id = ? ORDER BY exit_date ASC",
            (head['id'],)).fetchall()]
    except Exception as e:
        return unavailable(f"Could not read persisted trades: {e}", sid)
    finally:
        if conn is not None:
            conn.close()

    if not raw:
        return unavailable(
            "This backtest stored no trades (it predates trade persistence). "
            "Re-run it to enable real compliance checking.", sid)

    df = pd.DataFrame([{
        'entry_date': t.get('entry_date'),
        'exit_date': t.get('exit_date'),
        'entry_price': t.get('entry_price'),
        'exit_price': t.get('exit_price'),
        'size': t.get('size'),
        'symbol': t.get('symbol') or symbol or 'EUR-USD',
    } for t in raw])

    return rows_from_trades(
        df, phase=phase, account_sizes=account_sizes,
        strategy_id=sid, source=SOURCE_RESULTS_DB,
    )


# ==============================================================================
# CONVENIENCE: resolve the best available source
# ==============================================================================

def build_panel(
    live_trades=None,
    results_db_path: Optional[str] = None,
    decay_db_path: Optional[str] = None,
    strategy_id: str = '',
    strategy_name: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    phase: str = 'challenge',
    account_sizes: Optional[List[int]] = None,
) -> PanelResult:
    """
    Try trade sources in descending order of fidelity, then fall back to
    UNAVAILABLE -- never to a proxy.

        1. live_trades   in-process result           exact
        2. results_db    backtest_trades             exact (entry AND exit prices)
        3. decay_db      strategy_trades             exact for FX, approximate otherwise
        4. none          UNAVAILABLE with a reason
    """
    if live_trades is not None:
        n = len(live_trades) if hasattr(live_trades, '__len__') else 0
        if n:
            return rows_from_trades(
                live_trades, phase=phase, account_sizes=account_sizes,
                strategy_id=strategy_id, source=SOURCE_LIVE,
            )

    if results_db_path:
        r = rows_from_results_db(
            results_db_path, variant_id=strategy_id or None,
            strategy_name=strategy_name, symbol=symbol, timeframe=timeframe,
            phase=phase, account_sizes=account_sizes,
        )
        if r.available:
            return r
        first_reason = r.reason
    else:
        first_reason = None

    if decay_db_path and strategy_id:
        r = rows_from_decay_db(
            decay_db_path, strategy_id, symbol=symbol,
            phase=phase, account_sizes=account_sizes,
        )
        if r.available:
            return r
        if first_reason:
            r.reason = f"{first_reason} Decay DB: {r.reason}"
        return r

    if first_reason:
        return unavailable(first_reason, strategy_id)

    # Names the tables the caller has to go and look at. The previous wording
    # told them to apply apply_trade_persistence_patch.py, which no longer
    # exists -- the patch is applied and the script was removed. Advice that
    # points at a deleted file is worse than no advice: it sends someone
    # looking for the wrong thing.
    return unavailable(
        "No trade-level data. FTMO compliance needs individual trades with "
        "timestamps, not summary statistics. Check that backtest_results has "
        "a row for this strategy and that backtest_trades has rows for that "
        "backtest_id; if not, re-run the backtest, or persist trades via "
        "DecayCalculator.save_trades().",
        strategy_id,
    )


# ==============================================================================
# RENDER HELPERS (framework-agnostic -- return plain strings)
# ==============================================================================

def row_cells(row: dict) -> List[tuple]:
    """
    (label, is_pass) pairs for one account-size row, in table order.
    Includes Min Days, which the old table omitted.
    """
    return [
        ('PASS' if row['daily_ok'] else 'FAIL', row['daily_ok']),
        ('PASS' if row['total_ok'] else 'FAIL', row['total_ok']),
        ('PASS' if row['min_days_ok'] else 'FAIL', row['min_days_ok']),
        ('PASS' if row['profit_ok'] else 'FAIL', row['profit_ok']),
        ('PASS' if row['passed'] else 'FAIL', row['passed']),
    ]


def caption(result: PanelResult) -> str:
    """One-line provenance string. Always states where the numbers came from."""
    if not result.available:
        return f"FTMO compliance unavailable - {result.reason}"
    src = {SOURCE_LIVE: 'live backtest trades',
           SOURCE_RESULTS_DB: 'persisted trades (results DB, exact prices)',
           SOURCE_DECAY_DB: 'persisted trades (decay DB)'}.get(result.source, result.source)
    base = f"Real FTMOComplianceChecker output on {src}"
    if result.strategy_id:
        base += f" for {result.strategy_id}"
    if result.approximate:
        base += f" -- {result.approximate_note}"
    return base