# ==============================================================================
# prohibited_patterns.py
# ==============================================================================
# Phase 1, Item 8 -- prohibited-pattern detector.
#
# Most prop firms ban a specific set of trading behaviours outright. A strategy
# using them CANNOT BE FUNDED regardless of how well it performs, so this is a
# harder gate than any performance filter: a banned strategy with a Sharpe of 3
# is worth exactly zero, and every backtest hour spent on it is wasted.
#
# THIS PIPELINE ACTIVELY GENERATES THEM
# -------------------------------------
# mutation_config.POSITION_SIZING feeds the mutation prompt, and it lists:
#
#     Martingale (increase after loss)
#     Scale in (multiple entries)
#     DCA (dollar cost averaging)
#     Pyramid (add to winners)
#
# The first three are, in various combinations, exactly what firms prohibit.
# So this is not a hypothetical safeguard against sloppy authoring -- the
# mutation agent is being told to produce these. The detector is one half of
# the fix; the other half is removing the instruction (see the patcher).
#
# WHAT IS DETECTED
# ----------------
#   martingale        position size increases after a losing trade
#   grid              repeated entries at intervals while price moves against
#   hedging           simultaneous long and short in the same instrument
#   sub_threshold     positions held below a firm's minimum duration
#   overexposure      many simultaneous open positions (correlated-risk stacking)
#
# TWO LAYERS, same rationale as the lookahead detector:
#
#   scan_source()   AST. Milliseconds, catches obviously authored patterns.
#                   Can false-positive; cannot prove absence.
#   scan_trades()   Behavioural. Reads what the strategy ACTUALLY DID from a
#                   trade list. Cannot be evaded by clever code structure,
#                   because it looks at outcomes rather than syntax.
#
# scan_trades is the one that matters. A strategy can implement martingale
# without ever writing `* 2`, and static analysis will miss it every time.
# This layer became possible once trades were persisted to backtest_trades.
#
# FIRM VARIATION
# --------------
# Thresholds differ by firm. The defaults here are deliberately conservative.
# Once the target firm is confirmed, set them from its rulebook rather than
# trusting these -- and note that some firms permit hedging while others void
# the account for it. Nothing here is a substitute for reading the contract.
# ==============================================================================

import ast
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

CRITICAL = 'CRITICAL'
WARNING = 'WARNING'
INFO = 'INFO'
SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}

# Conservative defaults. Replace from the target firm's rulebook.
DEFAULTS = {
    'min_hold_seconds': 60,          # many firms ban sub-minute "tick scalping"
    'max_concurrent_positions': 5,
    'martingale_min_events': 3,      # size-ups after a loss before flagging
    'martingale_size_ratio': 1.5,    # what counts as "increased"
    'grid_min_legs': 4,              # adds against the position
    'sub_threshold_max_pct': 0.20,   # share of trades allowed under min hold
}


@dataclass
class PatternFinding:
    severity: str
    pattern: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    line: Optional[int] = None

    def __str__(self):
        loc = f" L{self.line}" if self.line else ""
        return f"  [{self.severity:8}] {self.pattern}{loc}\n             {self.detail}"


@dataclass
class PatternReport:
    name: str
    findings: List[PatternFinding] = field(default_factory=list)
    layer: str = ''
    trades_analysed: int = 0
    error: Optional[str] = None

    @property
    def critical(self) -> List[PatternFinding]:
        return [f for f in self.findings if f.severity == CRITICAL]

    @property
    def failed(self) -> bool:
        return bool(self.critical) or self.error is not None

    @property
    def patterns(self) -> set:
        return {f.pattern for f in self.findings}

    def summary(self) -> str:
        L = [f"\n{'=' * 68}", f"  PROHIBITED PATTERNS ({self.layer}): {self.name}", '=' * 68]
        if self.error:
            L += [f"  [ERROR] {self.error}", '=' * 68]
            return '\n'.join(L)
        if self.trades_analysed:
            L.append(f"  Trades analysed: {self.trades_analysed}")
        if not self.findings:
            L.append("  No prohibited patterns detected.")
            if self.layer == 'static':
                L.append("  (static scan only -- run scan_trades for behaviour)")
        else:
            for f in sorted(self.findings, key=lambda x: (SEVERITY_ORDER[x.severity], x.pattern)):
                L.append(str(f))
            L.append("")
            L.append(f"  VERDICT: {'FAIL - not fundable' if self.failed else 'PASS (with warnings)'}")
        L.append('=' * 68)
        return '\n'.join(L)


# ==============================================================================
# LAYER 1: STATIC
# ==============================================================================

class _Visitor(ast.NodeVisitor):
    MARTINGALE_NAMES = {'martingale', 'double_down', 'recover', 'recovery',
                        'loss_multiplier', 'after_loss'}
    GRID_NAMES = {'grid', 'grid_step', 'grid_levels', 'grid_spacing', 'averaging',
                  'dca', 'add_on_loss'}

    def __init__(self, src_lines):
        self.findings: List[PatternFinding] = []
        self.src = src_lines

    def _add(self, sev, pattern, node, detail):
        self.findings.append(PatternFinding(sev, pattern, detail, line=getattr(node, 'lineno', None)))

    def visit_Name(self, node):
        low = node.id.lower()
        if low in self.MARTINGALE_NAMES:
            self._add(CRITICAL, 'martingale', node,
                      f"Identifier '{node.id}' names a martingale mechanism. Most "
                      f"prop firms void accounts for size-up-after-loss.")
        elif low in self.GRID_NAMES:
            self._add(WARNING, 'grid', node,
                      f"Identifier '{node.id}' suggests grid or averaging-down. "
                      f"Confirm against the firm's rulebook.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        low = node.attr.lower()
        if low in self.MARTINGALE_NAMES:
            self._add(CRITICAL, 'martingale', node,
                      f"Attribute '{node.attr}' names a martingale mechanism.")
        elif low in self.GRID_NAMES:
            self._add(WARNING, 'grid', node,
                      f"Attribute '{node.attr}' suggests grid or averaging-down.")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        # size *= 2 style escalation
        if isinstance(node.op, ast.Mult) and self._is_size_target(node.target):
            v = self._const_num(node.value)
            if v is not None and v > 1:
                self._add(CRITICAL, 'martingale', node,
                          f"Position size multiplied by {v}. Escalating size is the "
                          f"defining feature of martingale.")
        self.generic_visit(node)

    @staticmethod
    def _const_num(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        return None

    @staticmethod
    def _is_size_target(t) -> bool:
        name = t.attr if isinstance(t, ast.Attribute) else (t.id if isinstance(t, ast.Name) else '')
        return any(k in name.lower() for k in ('size', 'stake', 'lot', 'qty', 'volume', 'units'))

    def visit_Call(self, node):
        # buy() and sell() both reachable in one next() -> potential hedging
        self.generic_visit(node)


def _hedging_in_source(tree) -> bool:
    """Both buy and sell reachable inside a single next() without a close()."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'next':
            calls = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    calls.add(n.func.attr)
            if {'buy', 'sell'} <= calls and 'close' not in calls:
                return True
    return False


def scan_source(source: str, name: str = '<string>') -> PatternReport:
    """AST scan. Fast, no data. May false-positive; cannot prove absence."""
    rep = PatternReport(name=name, layer='static')
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        rep.error = f"line {e.lineno}: {e.msg}"
        return rep

    v = _Visitor(source.splitlines())
    v.visit(tree)
    rep.findings = v.findings

    if _hedging_in_source(tree):
        rep.findings.append(PatternFinding(
            WARNING, 'hedging',
            "next() can call both buy() and sell() without close(), which can open "
            "opposing positions in the same instrument. Some firms void accounts "
            "for this; others permit it. Verify against the rulebook."))
    return rep


def scan_file(path: str) -> PatternReport:
    with open(path, 'r', encoding='utf-8') as f:
        return scan_source(f.read(), name=os.path.basename(path))


# ==============================================================================
# LAYER 2: BEHAVIOURAL -- what the strategy actually did
# ==============================================================================

def _norm(trades) -> pd.DataFrame:
    df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades or [])
    if df.empty:
        return df
    for c in ('entry_date', 'exit_date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    if 'pnl' not in df.columns and {'entry_price', 'exit_price', 'size'} <= set(df.columns):
        df['pnl'] = (df['exit_price'] - df['entry_price']) * df['size']
    if 'symbol' not in df.columns:
        df['symbol'] = 'UNKNOWN'
    return df.sort_values('entry_date').reset_index(drop=True)


def scan_trades(trades, name: str = '', thresholds: Optional[Dict] = None) -> PatternReport:
    """
    Behavioural scan of an actual trade list.

    This is the layer that matters. A strategy can implement martingale without
    ever writing `* 2`; static analysis will miss it every time. Outcomes cannot
    be disguised.
    """
    th = dict(DEFAULTS)
    th.update(thresholds or {})
    rep = PatternReport(name=name or 'trades', layer='behavioural')

    df = _norm(trades)
    if df.empty:
        rep.error = "No trades to analyse"
        return rep
    rep.trades_analysed = len(df)

    _check_martingale(df, th, rep)
    _check_hedging(df, th, rep)
    _check_sub_threshold(df, th, rep)
    _check_grid(df, th, rep)
    _check_overexposure(df, th, rep)
    return rep


def _check_martingale(df, th, rep):
    if 'size' not in df.columns or 'pnl' not in df.columns or len(df) < 3:
        return
    sizes = df['size'].abs().values
    pnls = df['pnl'].values

    ups_after_loss = 0
    ups_after_win = 0
    for i in range(1, len(df)):
        if sizes[i - 1] <= 0:
            continue
        ratio = sizes[i] / sizes[i - 1]
        if ratio >= th['martingale_size_ratio']:
            if pnls[i - 1] < 0:
                ups_after_loss += 1
            else:
                ups_after_win += 1

    if ups_after_loss >= th['martingale_min_events']:
        # Distinguish martingale from anti-martingale (pyramiding), which is
        # usually permitted: what matters is whether size-ups follow LOSSES.
        detail = (f"Position size increased by >={th['martingale_size_ratio']}x "
                  f"after a loss on {ups_after_loss} occasions "
                  f"(vs {ups_after_win} after wins). Sizing up into losses is "
                  f"martingale and is prohibited by most prop firms.")
        rep.findings.append(PatternFinding(
            CRITICAL, 'martingale', detail,
            {'size_ups_after_loss': ups_after_loss, 'size_ups_after_win': ups_after_win}))


def _check_hedging(df, th, rep):
    if 'size' not in df.columns or 'exit_date' not in df.columns:
        return
    events = 0
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('entry_date')
        for i, row in g.iterrows():
            overlap = g[(g['entry_date'] < row['exit_date']) &
                        (g['exit_date'] > row['entry_date']) &
                        (g.index != i)]
            if overlap.empty:
                continue
            if (np.sign(overlap['size']) != np.sign(row['size'])).any():
                events += 1
    if events:
        rep.findings.append(PatternFinding(
            CRITICAL, 'hedging',
            f"{events} overlapping position pair(s) held long and short in the same "
            f"instrument simultaneously. Many firms void accounts for this.",
            {'overlap_events': events}))


def _check_sub_threshold(df, th, rep):
    if not {'entry_date', 'exit_date'} <= set(df.columns):
        return
    dur = (df['exit_date'] - df['entry_date']).dt.total_seconds()
    dur = dur.dropna()
    if dur.empty:
        return
    short = int((dur < th['min_hold_seconds']).sum())
    pct = short / len(dur)
    if pct > th['sub_threshold_max_pct']:
        rep.findings.append(PatternFinding(
            CRITICAL, 'sub_threshold',
            f"{short}/{len(dur)} trades ({pct * 100:.0f}%) held under "
            f"{th['min_hold_seconds']}s. Firms that ban tick scalping typically "
            f"disallow this above a small share of trades.",
            {'short_trades': short, 'pct': pct}))
    elif short:
        rep.findings.append(PatternFinding(
            WARNING, 'sub_threshold',
            f"{short} trade(s) held under {th['min_hold_seconds']}s "
            f"({pct * 100:.1f}%) -- under the threshold, but worth confirming.",
            {'short_trades': short, 'pct': pct}))


def _check_grid(df, th, rep):
    """
    Grid / averaging-down: repeatedly adding to a position while price moves
    against it. Distinct from pyramiding, which adds while price moves FOR.
    """
    if not {'entry_price', 'size', 'entry_date'} <= set(df.columns):
        return
    worst = 0
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('entry_date').reset_index(drop=True)
        run = 0
        for i in range(1, len(g)):
            prev, cur = g.iloc[i - 1], g.iloc[i]
            if np.sign(cur['size']) != np.sign(prev['size']):
                run = 0
                continue
            # Adding while price moved against the existing direction
            adverse = ((cur['entry_price'] < prev['entry_price']) if cur['size'] > 0
                       else (cur['entry_price'] > prev['entry_price']))
            overlapping = ('exit_date' in g.columns and
                           pd.notna(prev.get('exit_date')) and
                           cur['entry_date'] < prev['exit_date'])
            run = run + 1 if (adverse and overlapping) else 0
            worst = max(worst, run)

    if worst >= th['grid_min_legs']:
        rep.findings.append(PatternFinding(
            CRITICAL, 'grid',
            f"Up to {worst + 1} consecutive entries added while price moved against "
            f"an open position (averaging down). Grid and martingale-style "
            f"recovery are prohibited by most firms.",
            {'max_adverse_legs': worst + 1}))


def _check_overexposure(df, th, rep):
    if not {'entry_date', 'exit_date'} <= set(df.columns):
        return
    events = []
    for _, row in df.iterrows():
        n = int(((df['entry_date'] < row['exit_date']) &
                 (df['exit_date'] > row['entry_date'])).sum())
        events.append(n)
    peak = max(events) if events else 0
    if peak > th['max_concurrent_positions']:
        rep.findings.append(PatternFinding(
            WARNING, 'overexposure',
            f"Up to {peak} positions open simultaneously (limit "
            f"{th['max_concurrent_positions']}). Firms cap correlated exposure, "
            f"and concentration is what breaches a daily-loss rule.",
            {'peak_concurrent': peak}))


# ==============================================================================
# COMBINED GATE
# ==============================================================================

def gate(source: Optional[str] = None, trades=None, name: str = 'strategy',
         thresholds: Optional[Dict] = None, verbose: bool = False) -> bool:
    """
    True if the strategy may proceed.

    Both layers run when available; the behavioural one is authoritative.
    A strategy with no trades is not cleared -- it is simply unproven, and
    scan_source alone cannot establish absence.
    """
    ok = True
    if source is not None:
        r = scan_source(source, name=name)
        if verbose:
            print(r.summary())
        ok = ok and not r.failed
    if trades is not None:
        r = scan_trades(trades, name=name, thresholds=thresholds)
        if verbose:
            print(r.summary())
        ok = ok and not r.failed
    return ok
