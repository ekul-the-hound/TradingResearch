# ==============================================================================
# dashboard_portfolio_panel.py
# ==============================================================================
# Data layer for two dashboard features:
#
#   1. SIDE-BY-SIDE COMPARISON -- N results in aligned columns, with a delta
#      column showing portfolio-minus-best-constituent. The interesting number
#      is not "did the portfolio do well" but "did combining help, or did it
#      just average."
#
#   2. FIRM RULES FORM -- editable thresholds, and capability toggles that are
#      LOCKED when no implementation backs them. A control you cannot tick is
#      the UI expression of firm_rules.IMPLEMENTED.
#
# Follows the dashboard_ftmo_panel.py convention: no reactpy import, no theme
# import, no rendering. Returns plain dataclasses; react_dashboard2.py draws
# them. That keeps the import graph acyclic and this module unit-testable
# without a browser.
#
# The refusal discipline carries over. A strategy that cannot be compared shows
# as unavailable WITH A REASON. It never shows as zeros, and it never quietly
# drops out of the table -- a missing column is indistinguishable from a column
# that was never requested.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import firm_rules
    from firm_rules import Capability, FirmRules, IMPLEMENTED, CAPABILITY_NOTES
    _RULES_OK = True
except Exception as _e:                                   # pragma: no cover
    _RULES_OK = False
    _RULES_ERR = str(_e)

try:
    import portfolio_merge
    from portfolio_merge import PortfolioMergeError
    _MERGE_OK = True
except Exception as _e:                                   # pragma: no cover
    _MERGE_OK = False
    _MERGE_ERR = str(_e)


# ==============================================================================
# COMPARISON
# ==============================================================================

# (attribute, label, format spec, higher_is_better)
# higher_is_better=None means the metric is descriptive, not scored -- no delta
# arrow is drawn, because an arrow implies a judgement the metric cannot carry.
METRIC_SPEC: List[Tuple[str, str, str, Optional[bool]]] = [
    ('total_return_pct',  'Total Return',   '{:+.2f}%',  True),
    ('sharpe_ratio',      'Sharpe',         '{:.2f}',    True),
    ('max_drawdown_pct',  'Max Drawdown',   '{:.2f}%',   False),
    ('win_rate',          'Win Rate',       '{:.1f}%',   True),
    ('profit_factor',     'Profit Factor',  '{:.2f}',    True),
    ('total_trades',      'Trades',         '{:.0f}',    None),
    ('trades_per_day',    'Trades/Day',     '{:.2f}',    None),
    ('avg_trade_return_pct', 'Avg Trade',   '{:+.3f}%',  True),
]


@dataclass
class MetricCell:
    """One metric for one column."""
    raw: Optional[float]
    text: str
    available: bool = True

    @classmethod
    def missing(cls, why: str = 'n/a') -> "MetricCell":
        return cls(raw=None, text=why, available=False)


@dataclass
class ComparisonColumn:
    """One strategy or portfolio in the comparison table."""
    key: str
    label: str
    is_portfolio: bool = False
    available: bool = True
    reason: str = ''
    returns_source: str = ''
    cells: Dict[str, MetricCell] = field(default_factory=dict)
    member_ids: List[str] = field(default_factory=list)

    @property
    def provenance_ok(self) -> bool:
        """True only if the numbers came from executed trades."""
        return self.returns_source == 'trade_list'


@dataclass
class DeltaCell:
    """Portfolio value minus the best individual value, for one metric."""
    raw: Optional[float]
    text: str
    better: Optional[bool] = None      # None = not scored / unavailable


@dataclass
class ComparisonTable:
    columns: List[ComparisonColumn] = field(default_factory=list)
    metric_keys: List[str] = field(default_factory=list)
    metric_labels: Dict[str, str] = field(default_factory=dict)
    deltas: Dict[str, DeltaCell] = field(default_factory=dict)
    delta_baseline: str = ''
    notes: List[str] = field(default_factory=list)

    @property
    def has_portfolio(self) -> bool:
        return any(c.is_portfolio for c in self.columns)

    @property
    def usable_columns(self) -> List[ComparisonColumn]:
        return [c for c in self.columns if c.available]


def _fmt(value: Optional[float], spec: str) -> str:
    if value is None:
        return '--'
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _column_from_result(result, key: str, label: str = '',
                        is_portfolio: bool = False) -> ComparisonColumn:
    col = ComparisonColumn(
        key=key,
        label=label or getattr(result, 'strategy_name', '') or key,
        is_portfolio=is_portfolio,
        returns_source=getattr(result, 'returns_source', '') or '',
    )

    params = getattr(result, 'strategy_params', None) or {}
    if isinstance(params, dict):
        col.member_ids = list(params.get('members', []) or [])

    for attr, _label, spec, _hib in METRIC_SPEC:
        val = getattr(result, attr, None)
        if val is None:
            col.cells[attr] = MetricCell.missing()
        else:
            col.cells[attr] = MetricCell(raw=float(val), text=_fmt(val, spec))
    return col


def unavailable_column(key: str, reason: str,
                       label: str = '') -> ComparisonColumn:
    """
    A column that could not be built.

    It still occupies a slot in the table. Silently omitting it would make
    "this strategy failed to load" look identical to "you didn't select it".
    """
    col = ComparisonColumn(key=key, label=label or key,
                           available=False, reason=reason)
    for attr, _l, _s, _h in METRIC_SPEC:
        col.cells[attr] = MetricCell.missing('--')
    return col


def build_comparison(
    results: Sequence[Any],
    portfolio: Optional[Any] = None,
    labels: Optional[Dict[str, str]] = None,
) -> ComparisonTable:
    """
    Assemble the side-by-side table.

    Args:
        results:   individual CanonicalResults.
        portfolio: an optional merged CanonicalResult (from
                   portfolio_merge.merge_strategies(...).canonical). Because
                   the merge emits an ordinary CanonicalResult, it renders
                   through this same path with no special casing.
        labels:    optional display-name override keyed by strategy_id.
    """
    labels = labels or {}
    table = ComparisonTable(
        metric_keys=[a for a, _l, _s, _h in METRIC_SPEC],
        metric_labels={a: l for a, l, _s, _h in METRIC_SPEC},
    )

    for r in results:
        sid = getattr(r, 'strategy_id', '') or getattr(r, 'strategy_name', '')
        if not sid:
            table.columns.append(unavailable_column(
                key=f'unnamed_{len(table.columns)}',
                reason='result has no strategy_id'))
            continue
        table.columns.append(
            _column_from_result(r, sid, labels.get(sid, ''), False))

    if portfolio is not None:
        pid = getattr(portfolio, 'strategy_id', 'portfolio')
        table.columns.append(
            _column_from_result(portfolio, pid, labels.get(pid, 'Portfolio'),
                                True))
        _attach_deltas(table)

    # Provenance warnings -- surfaced, not silently tolerated.
    for c in table.columns:
        if c.available and c.returns_source and not c.provenance_ok:
            table.notes.append(
                f"{c.label}: returns_source='{c.returns_source}', not derived "
                f"from executed trades. Metrics are not comparable."
            )

    return table


def _attach_deltas(table: ComparisonTable) -> None:
    """
    Portfolio minus the BEST individual, per metric.

    Best, not mean. Comparing a portfolio to the average of its members
    flatters it: any combination beats its own worst member. The question
    worth answering is whether the portfolio beats the single best thing you
    could have run instead.
    """
    pcol = next((c for c in table.columns if c.is_portfolio), None)
    individuals = [c for c in table.columns
                   if not c.is_portfolio and c.available]
    if pcol is None or not individuals:
        return

    for attr, _label, spec, hib in METRIC_SPEC:
        pcell = pcol.cells.get(attr)
        if pcell is None or not pcell.available or pcell.raw is None:
            table.deltas[attr] = DeltaCell(raw=None, text='--')
            continue

        # Built as a concrete List[float] rather than a comprehension: the
        # comprehension's `raw is not None` guard tests a re-accessed
        # attribute, so a type checker cannot narrow the element type and
        # max()/min() end up receiving list[float | None].
        vals: List[float] = []
        for c in individuals:
            cell = c.cells.get(attr)
            if cell is not None and cell.available and cell.raw is not None:
                vals.append(float(cell.raw))
        if not vals:
            table.deltas[attr] = DeltaCell(raw=None, text='--')
            continue

        if hib is None:
            table.deltas[attr] = DeltaCell(raw=None, text='--')
            continue

        best = max(vals) if hib else min(vals)
        diff = pcell.raw - best
        better = (diff > 0) if hib else (diff < 0)
        if abs(diff) < 1e-12:
            better = None

        sign_spec = spec if spec.startswith('{:+') else spec.replace('{:', '{:+')
        table.deltas[attr] = DeltaCell(
            raw=diff, text=_fmt(diff, sign_spec), better=better)

    table.delta_baseline = 'best individual strategy'


def comparison_caption(table: ComparisonTable) -> str:
    """One-line provenance string, mirroring dashboard_ftmo_panel.caption."""
    if not table.columns:
        return 'No strategies selected.'
    n_ok = len(table.usable_columns)
    n_bad = len(table.columns) - n_ok
    base = f"{n_ok} result(s) compared"
    if table.has_portfolio:
        base += f"; delta is portfolio minus {table.delta_baseline}"
    if n_bad:
        base += f"; {n_bad} could not be loaded"
    return base + '.'


# ==============================================================================
# FIRM RULES FORM
# ==============================================================================

# (field, label, kind, help text)
NUMERIC_FIELDS: List[Tuple[str, str, str, str]] = [
    ('firm_name',              'Firm name',           'text',
     'Display only.'),
    ('max_daily_loss_pct',     'Max daily loss',      'pct',
     'Fraction of the anchor balance. 5% is 0.05.'),
    ('max_total_drawdown_pct', 'Max total drawdown',  'pct',
     'Fraction of the initial balance.'),
    ('min_trading_days',       'Min trading days',    'int',
     'Distinct days with at least one trade.'),
    ('max_calendar_days',      'Max calendar days',   'int_or_none',
     'Blank for no time limit.'),
    ('consistency_max_day_pct', 'Consistency cap',    'pct_or_none',
     'Largest share of total profit one day may contribute. '
     'NOT YET ENFORCED -- setting it flags results as partial.'),
    ('reset_timezone',         'Daily reset timezone', 'text',
     'Where the firm\'s trading day starts. Affects which day a trade '
     'lands on.'),
]


@dataclass
class FormField:
    name: str
    label: str
    kind: str
    value: Any
    help: str = ''
    error: str = ''


@dataclass
class CapabilityToggle:
    """
    A rule semantic in the UI.

    `locked` is the whole point. An unimplemented capability renders as a
    disabled control with the reason attached, so the form cannot be used to
    describe a firm the engine would then silently fail to check.
    """
    capability: str
    label: str
    enabled: bool
    locked: bool
    reason: str = ''
    group: str = ''


CAPABILITY_GROUPS: Dict[str, str] = {
    'static_drawdown': 'drawdown',
    'trailing_drawdown_intraday': 'drawdown',
    'trailing_drawdown_eod': 'drawdown',
    'daily_loss_includes_floating': 'daily_loss',
    'daily_loss_closed_only': 'daily_loss',
}

CAPABILITY_LABELS: Dict[str, str] = {
    'static_drawdown': 'Static drawdown (fixed floor)',
    'trailing_drawdown_intraday': 'Trailing drawdown (intraday high-water)',
    'trailing_drawdown_eod': 'Trailing drawdown (end-of-day ratchet)',
    'daily_loss_includes_floating': 'Daily loss includes floating P&L',
    'daily_loss_closed_only': 'Daily loss counts closed trades only',
    'min_trading_days': 'Minimum trading days',
    'max_calendar_days': 'Maximum calendar days',
    'consistency_rule': 'Consistency rule (per-day profit cap)',
    'weekend_holding_ban': 'No weekend holding',
    'news_trading_ban': 'No trading around news',
    'max_lot_size': 'Maximum lot size',
    'stop_loss_mandatory': 'Stop loss required on every trade',
}


def build_firm_form(rules: "FirmRules") -> Tuple[List[FormField],
                                                 List[CapabilityToggle]]:
    """Descriptors for rendering the firm-rules editor."""
    if not _RULES_OK:                                     # pragma: no cover
        raise RuntimeError(f"firm_rules unavailable: {_RULES_ERR}")

    fields = [
        FormField(name=n, label=l, kind=k, value=getattr(rules, n, None),
                  help=h)
        for n, l, k, h in NUMERIC_FIELDS
    ]

    active = {c.value for c in rules.required_capabilities}
    toggles = []
    for cap in Capability:
        locked = cap not in IMPLEMENTED
        toggles.append(CapabilityToggle(
            capability=cap.value,
            label=CAPABILITY_LABELS.get(cap.value) or str(cap.value),
            enabled=cap.value in active,
            locked=locked,
            reason=CAPABILITY_NOTES.get(cap, '') if locked else '',
            group=CAPABILITY_GROUPS.get(cap.value, ''),
        ))
    return fields, toggles


def _coerce(kind: str, raw: Any) -> Any:
    if kind == 'text':
        return str(raw)

    # Checked inline rather than via a `blank` variable: narrowing does not
    # propagate through an intermediate bool, so float(raw) below would still
    # be seen as possibly receiving None.
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if kind in ('int_or_none', 'pct_or_none'):
            return None
        # A blank on a REQUIRED field is a user error, not a missing value.
        # Raising routes it to the field's own error slot via apply_firm_form
        # instead of letting float(None) surface as a bare TypeError.
        raise ValueError(f"{kind} field cannot be empty")

    if kind in ('int', 'int_or_none'):
        return int(float(raw))
    return float(raw)


def apply_firm_form(
    payload: Dict[str, Any],
    capabilities: Optional[Sequence[str]] = None,
    base: Optional["FirmRules"] = None,
) -> Tuple[Optional["FirmRules"], List[FormField]]:
    """
    Build a FirmRules from submitted form values.

    Returns (rules, fields). On failure `rules` is None and the offending
    FormField carries `error`. Validation lives in FirmRules itself; this only
    coerces types and routes the message back to the right control, so the
    dashboard and any scripted caller reject the same profiles.
    """
    if not _RULES_OK:                                     # pragma: no cover
        raise RuntimeError(f"firm_rules unavailable: {_RULES_ERR}")

    # `base if base is not None else ...` rather than `base or ...`: the
    # explicit check narrows away the Optional, so the attribute access below
    # is provably safe rather than merely safe in practice.
    profile = base if base is not None else firm_rules.ftmo()
    kwargs: Dict[str, Any] = {
        'profit_targets': dict(profile.profit_targets),
        'account_sizes': list(profile.account_sizes),
    }
    fields: List[FormField] = []
    failed = False

    for name, label, kind, helptext in NUMERIC_FIELDS:
        raw = payload.get(name, getattr(profile, name, None))
        f = FormField(name=name, label=label, kind=kind, value=raw,
                      help=helptext)
        try:
            kwargs[name] = _coerce(kind, raw)
        except (TypeError, ValueError):
            f.error = f"'{raw}' is not a valid {kind.replace('_', ' ')}."
            failed = True
        fields.append(f)

    if capabilities is not None:
        try:
            kwargs['required_capabilities'] = [Capability(c)
                                               for c in capabilities]
        except ValueError as e:
            fields.append(FormField(name='required_capabilities',
                                    label='Capabilities', kind='caps',
                                    value=list(capabilities), error=str(e)))
            failed = True

    if failed:
        return None, fields

    try:
        return FirmRules(**kwargs), fields
    except ValueError as e:
        msg = str(e)
        target = next((f for f in fields if f.name in msg), None)
        if target is None:
            # Cross-field failures (e.g. daily limit above total) belong on
            # the control the user most likely just changed.
            target = next((f for f in fields
                           if f.name == 'max_daily_loss_pct'), fields[0])
        target.error = msg
        return None, fields


def firm_status_line(rules: "FirmRules") -> Dict[str, Any]:
    """Badge payload for the top of the firm-rules panel."""
    gaps = rules.unsupported()
    return {
        'firm': rules.firm_name,
        'complete': not gaps,
        'n_unchecked': len(gaps),
        'unchecked': [g.capability.value for g in gaps],
        'text': rules.caveat_line(),
        'tone': 'green' if not gaps else 'amber',
    }


# ==============================================================================
# LOADING FROM THE RESULTS DB
# ==============================================================================
# Mirrors dashboard_ftmo_panel.rows_from_results_db: backtest_trades is the
# highest-fidelity persisted source, carrying entry AND exit prices plus pnl.
#
# The important design point is `has_trades` on the candidate list. A backtest
# that predates trade persistence cannot enter a portfolio merge, and the user
# needs to know that when PICKING strategies -- not after assembling a
# selection and getting a refusal. Surfacing the constraint at selection time
# is the difference between a guard rail and a trapdoor.

import os
import sqlite3


def _as_float(value: Any, default: float) -> float:
    """
    Coerce a SQLite value to float, falling back on NULL or garbage.

    Only ever used for fields CanonicalResult declares non-Optional. Fields
    that are genuinely Optional keep their None and never come through here.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def list_candidates(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Selectable backtests, newest first, each flagged with whether it has a
    persisted trade ledger.
    """
    if not db_path or not os.path.exists(db_path):
        return []

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row

        has_table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='backtest_trades'"
        ).fetchone() is not None

        rows = [dict(r) for r in conn.execute(
            "SELECT id, strategy_name, variant_id, symbol, timeframe, "
            "total_return_pct, sharpe_ratio, total_trades "
            "FROM backtest_results ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]

        counts: Dict[int, int] = {}
        if has_table and rows:
            ids = tuple(r['id'] for r in rows)
            marks = ','.join('?' * len(ids))
            for cr_ in conn.execute(
                f"SELECT backtest_id, COUNT(*) n FROM backtest_trades "
                f"WHERE backtest_id IN ({marks}) GROUP BY backtest_id", ids
            ).fetchall():
                counts[cr_['backtest_id']] = cr_['n']
    except Exception:
        return []
    finally:
        if conn is not None:
            conn.close()

    out = []
    for r in rows:
        n = counts.get(r['id'], 0)
        out.append({
            'id': r['id'],
            'key': r['variant_id'] or r['strategy_name'] or f"bt{r['id']}",
            'label': r['variant_id'] or r['strategy_name'] or f"bt{r['id']}",
            'symbol': r['symbol'],
            'timeframe': r['timeframe'],
            'total_return_pct': r['total_return_pct'],
            'sharpe_ratio': r['sharpe_ratio'],
            'n_trades_persisted': n,
            'has_trades': n > 0,
            'blocked_reason': '' if n > 0 else
                'No persisted trades -- re-run this backtest to enable merging.',
        })
    return out


def load_result(db_path: str, backtest_id: int):
    """
    Build a CanonicalResult with a real trade ledger.

    Returns (result, reason). On failure result is None and reason explains
    why, so the caller can render an unavailable_column rather than dropping
    the strategy.
    """
    from canonical_result import CanonicalResult

    if not db_path or not os.path.exists(db_path):
        return None, 'Results database not found'

    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        head = conn.execute(
            "SELECT * FROM backtest_results WHERE id = ?", (backtest_id,)
        ).fetchone()
        if head is None:
            return None, f'No backtest with id {backtest_id}'
        head = dict(head)

        has_table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='backtest_trades'"
        ).fetchone() is not None
        if not has_table:
            return None, ('backtest_trades table does not exist; this '
                          'database predates trade persistence')

        trades = [dict(t) for t in conn.execute(
            "SELECT * FROM backtest_trades WHERE backtest_id = ? "
            "ORDER BY exit_date ASC", (backtest_id,)
        ).fetchall()]
    except Exception as e:
        return None, f'Could not read backtest {backtest_id}: {e}'
    finally:
        if conn is not None:
            conn.close()

    sid = head.get('variant_id') or head.get('strategy_name') or f'bt{backtest_id}'

    if not trades:
        return None, (f"'{sid}' stored no trades (predates trade "
                      f"persistence). Re-run it to enable merging.")

    trade_rows = [{
        'entry_date': t.get('entry_date'),
        'exit_date': t.get('exit_date'),
        'entry_price': t.get('entry_price'),
        'exit_price': t.get('exit_price'),
        'size': t.get('size'),
        'symbol': t.get('symbol') or head.get('symbol') or 'UNKNOWN',
        'pnl': t.get('pnl'),
    } for t in trades]

    starting = _as_float(head.get('initial_cash'), 100_000.0)

    # ending_value is non-Optional on CanonicalResult, so a NULL final_value
    # has to become SOMETHING. Falling back to the dataclass default would
    # invent an account balance. Summing the real persisted P&L instead keeps
    # the number derived from data that actually exists.
    final_value = head.get('final_value')
    if final_value is None:
        ending = starting + sum(_as_float(t.get('pnl'), 0.0) for t in trades)
    else:
        ending = _as_float(final_value, starting)

    result = CanonicalResult(
        strategy_id=sid,
        strategy_name=head.get('strategy_name') or sid,
        symbol=head.get('symbol') or '',
        timeframe=head.get('timeframe') or '',
        # Required fields: coerce, mirroring CanonicalResult.from_backtest.
        total_return_pct=_as_float(head.get('total_return_pct'), 0.0),
        total_trades=_as_int(head.get('total_trades'), len(trade_rows)),
        starting_value=starting,
        ending_value=ending,
        # Optional fields: pass None through untouched. None means unmeasured
        # and is deliberately distinct from 0.0 -- coercing these would be the
        # exact confident-wrong-number failure the audit removed.
        sharpe_ratio=head.get('sharpe_ratio'),
        max_drawdown_pct=head.get('max_drawdown_pct'),
        win_rate=head.get('win_rate'),
        profit_factor=head.get('profit_factor'),
        start_date=head.get('start_date') or '',
        end_date=head.get('end_date') or '',
        trade_list=trade_rows,
    )

    # CanonicalResult has no __post_init__; without this the result carries
    # returns_source='none' despite a complete ledger.
    try:
        result._compute_arrays()
    except Exception:
        pass

    return result, ''


def load_selection(db_path: str, backtest_ids: Sequence[int]):
    """
    Load several at once.

    Returns (results, failures) where failures is a list of
    (id, reason) so every selected item is accounted for in the UI.
    """
    results, failures = [], []
    for bid in backtest_ids:
        r, why = load_result(db_path, bid)
        if r is None:
            failures.append((bid, why))
        else:
            results.append(r)
    return results, failures


# ==============================================================================
# MERGE HELPER
# ==============================================================================

def try_merge(results: Sequence[Any], rules: "FirmRules",
              account_size: float = 100_000.0,
              weights: Optional[Dict[str, float]] = None,
              overlap: str = 'intersection') -> Dict[str, Any]:
    """
    Run the merge and normalise both outcomes into one render payload.

    A failed merge is a first-class result with a reason attached, not an
    exception the page has to catch and turn into a blank panel.
    """
    if not _MERGE_OK:                                     # pragma: no cover
        return {'ok': False, 'reason': f"portfolio_merge unavailable: {_MERGE_ERR}"}

    try:
        res = portfolio_merge.merge_strategies(
            results, rules=rules, account_size=account_size,
            weights=weights, overlap=overlap)
    except PortfolioMergeError as e:
        return {'ok': False, 'reason': str(e)}
    except Exception as e:                                # pragma: no cover
        return {'ok': False, 'reason': f"{type(e).__name__}: {e}"}

    d = res.diagnostics
    return {
        'ok': True,
        'result': res,
        'canonical': res.canonical,
        'summary': d.summary(),
        'warnings': list(d.warnings),
        'unchecked': [u.capability.value for u in d.unsupported_rules],
        'same_day_loss_days': d.same_day_loss_days,
        'worst_day_pct': d.worst_combined_day_pct,
        'worst_day_date': d.worst_combined_day_date,
        'window': (d.window_start, d.window_end),
        'dropped_pct': d.trades_dropped_pct,
    }