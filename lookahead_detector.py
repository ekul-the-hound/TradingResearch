# ==============================================================================
# lookahead_detector.py
# ==============================================================================
# Phase 1, Item 6 -- automated lookahead detector.
#
# A strategy has lookahead bias if a decision it makes at time t depends on data
# from after t. Such a strategy backtests beautifully and loses money live,
# which makes it the single most expensive class of bug in this pipeline: it
# does not announce itself, it inflates exactly the metrics used for promotion,
# and every hour spent optimising an infected strategy is wasted.
#
# This runs as a GATE -- before evaluation, not after -- so it is built in two
# layers with very different costs:
#
#   LAYER 1  scan_source()      pure AST, milliseconds, no data, no backtest.
#                               Catches the common authored mistakes. Cheap
#                               enough to run on every generated variant.
#
#   LAYER 2  perturbation_test() ground truth. Runs the strategy twice on data
#                               that is IDENTICAL up to a cut point and wildly
#                               different after it. Any entry decision before
#                               the cut that changes can only have come from
#                               future data. Costs a few backtests.
#
# Layer 1 can produce false positives and cannot prove absence. Layer 2 cannot
# produce false positives but is slower and only exercises the code paths the
# data actually triggers. They are complementary; run 1 on everything and 2 on
# anything that survives to evaluation.
#
# BACKTRADER INDEXING -- the reason layer 1 works at all
# -----------------------------------------------------
# In Backtrader, line objects are indexed relative to the current bar:
#     self.data.close[0]    current bar        legal
#     self.data.close[-1]   previous bar       legal
#     self.data.close[1]    NEXT BAR           lookahead
# A positive integer index is a direct read of the future. It is easy to write
# by accident when coming from pandas, where [1] means "second row".
#
# USAGE
#     from lookahead_detector import LookaheadDetector
#     d = LookaheadDetector()
#
#     report = d.scan_source(open('variant_07.py').read(), name='variant_07')
#     if report.failed:
#         print(report.summary())
#
#     result = d.perturbation_test(StrategyClass, price_df)
#     if not result.clean:
#         print(result.summary())
# ==============================================================================

import ast
import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ==============================================================================
# SEVERITY
# ==============================================================================

CRITICAL = 'CRITICAL'   # reads the future; strategy must not be evaluated
WARNING = 'WARNING'     # suspicious; needs a human look
INFO = 'INFO'           # worth knowing, usually fine

SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}


@dataclass
class Finding:
    severity: str
    rule: str
    line: int
    code: str
    message: str

    def __str__(self):
        return f"  [{self.severity:8}] L{self.line:<4} {self.rule}\n" \
               f"             {self.code.strip()}\n" \
               f"             {self.message}"


@dataclass
class ScanReport:
    name: str
    findings: List[Finding] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def critical(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == CRITICAL]

    @property
    def failed(self) -> bool:
        """A strategy fails the gate on any CRITICAL finding, or if it won't parse."""
        return bool(self.critical) or self.parse_error is not None

    def summary(self) -> str:
        lines = [f"\n{'=' * 68}", f"  LOOKAHEAD SCAN: {self.name}", '=' * 68]
        if self.parse_error:
            lines.append(f"  [CRITICAL] Could not parse: {self.parse_error}")
            lines.append('=' * 68)
            return '\n'.join(lines)
        if not self.findings:
            lines.append("  No lookahead patterns found (static scan only --")
            lines.append("  this does not prove absence; run perturbation_test).")
        else:
            for f in sorted(self.findings, key=lambda x: (SEVERITY_ORDER[x.severity], x.line)):
                lines.append(str(f))
            lines.append("")
            lines.append(f"  {len(self.critical)} critical, "
                         f"{len(self.findings) - len(self.critical)} other")
            lines.append(f"  VERDICT: {'FAIL - do not evaluate' if self.failed else 'PASS (with warnings)'}")
        lines.append('=' * 68)
        return '\n'.join(lines)


# ==============================================================================
# LAYER 1: STATIC SCAN
# ==============================================================================

class _Visitor(ast.NodeVisitor):
    """Walks a strategy's AST looking for known future-reading patterns."""

    # Backtrader line-like attribute names. A positive subscript on any of
    # these reads a future bar.
    LINE_NAMES = {
        'close', 'open', 'high', 'low', 'volume', 'openinterest',
        'datetime', 'data', 'data0', 'data1', 'dataclose', 'dataopen',
        'datahigh', 'datalow', 'datavolume', 'lines', 'l',
    }

    # Broker settings that fill orders using the bar the decision was made on.
    CHEAT_CALLS = {'set_coc', 'set_coo', 'set_cheat_on_close', 'set_cheat_on_open'}

    # pandas idioms that pull future rows backwards.
    PANDAS_FUTURE_METHODS = {'shift', 'diff'}

    def __init__(self, source_lines):
        self.findings: List[Finding] = []
        self.src = source_lines

    def _code(self, node):
        idx = node.lineno - 1
        return self.src[idx] if 0 <= idx < len(self.src) else ''

    def _add(self, sev, rule, node, msg):
        self.findings.append(Finding(sev, rule, node.lineno, self._code(node), msg))

    # Reaching back further than this without a length guard is suspicious.
    DEEP_LOOKBACK = 20

    # -- positive line indexing: self.data.close[1] ------------------------
    def visit_Subscript(self, node):
        idx_val = self._const_int(node.slice)
        if idx_val is None or not self._is_line_like(node.value):
            self.generic_visit(node)
            return

        if idx_val > 0:
            self._add(
                CRITICAL, 'positive-line-index', node,
                f"Index [{idx_val}] reads {idx_val} bar(s) into the FUTURE. "
                f"In Backtrader [0] is the current bar and [-{idx_val}] is the past."
            )
        elif idx_val <= -self.DEEP_LOOKBACK:
            # Backtrader preloads the whole series and resolves line[ago] as
            # array[idx + ago]. Early in the run, idx + ago goes NEGATIVE, and
            # Python then wraps it to the END of the array -- returning FUTURE
            # data with no exception and no warning.
            #
            # Verified: on a strictly increasing 80-bar series, close[-50] at
            # bar 0 returns the value from bar 30. Thirty bars ahead, silently.
            #
            # Legal once enough bars have elapsed, so this is a warning rather
            # than a rejection -- but it needs a `if len(self) < N: return`
            # guard, and the perturbation layer will confirm either way.
            self._add(
                WARNING, 'unguarded-deep-lookback', node,
                f"Index [{idx_val}] reaches back {abs(idx_val)} bars. Before bar "
                f"{abs(idx_val)} this wraps to the END of the preloaded series and "
                f"returns future data silently. Guard with `if len(self) < "
                f"{abs(idx_val)}: return`."
            )
        self.generic_visit(node)

    @staticmethod
    def _const_int(slice_node):
        # NOTE: an ast.Index unwrapping branch used to sit here for py<3.9.
        # Since 3.9 ast.parse never emits Index nodes (verified on 3.12: a
        # subscript slice comes back as Constant), so the branch was dead --
        # and ast.Index subclasses slice, which has no .value, so the code in
        # it could not have run correctly anyway. Removed rather than silenced.
        n = slice_node
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            # Python parses -50 as UnaryOp(USub, Constant(50)), never as a
            # negative Constant. Omitting USub here meant every negative index
            # resolved to None and was silently skipped -- which made the
            # "negative indexing is fine" tests pass for the wrong reason and
            # left unguarded-deep-lookback unreachable.
            inner = n.operand
            if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
                return -inner.value if isinstance(n.op, ast.USub) else inner.value
        return None

    def _is_line_like(self, node) -> bool:
        """True for self.data.close, self.dataclose, self.fast_ma, self.lines.x ..."""
        if isinstance(node, ast.Attribute):
            if node.attr in self.LINE_NAMES:
                return True
            # self.<something>_ma / self.rsi etc. -- attribute on self is very
            # likely an indicator line in a bt.Strategy.
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                return True
            return self._is_line_like(node.value)
        if isinstance(node, ast.Call):
            return self._is_line_like(node.func)
        return False

    # -- calls: .get(ago=+n), cheat-on-close, pandas shift(-n) -------------
    def visit_Call(self, node):
        fn = node.func

        if isinstance(fn, ast.Attribute):
            # cerebro.broker.set_coc(True)
            if fn.attr in self.CHEAT_CALLS:
                truthy = any(isinstance(a, ast.Constant) and a.value is True for a in node.args)
                if truthy or not node.args:
                    self._add(
                        CRITICAL, 'cheat-on-close-or-open', node,
                        f"{fn.attr}() fills orders using the same bar the decision was "
                        f"made on. Backtests with this enabled are not reproducible live."
                    )

            # line.get(ago=1, size=n) -- positive ago is the future
            if fn.attr == 'get':
                for kw in node.keywords:
                    if kw.arg == 'ago':
                        v = self._const_int(kw.value)
                        if v is not None and v > 0:
                            self._add(CRITICAL, 'positive-ago', node,
                                      f"get(ago={v}) reads {v} bar(s) ahead.")

            # df['x'].shift(-1) -- pulls a future row into the present
            if fn.attr in self.PANDAS_FUTURE_METHODS:
                for a in list(node.args) + [k.value for k in node.keywords]:
                    v = self._negative_int(a)
                    if v is not None:
                        self._add(CRITICAL, 'negative-shift', node,
                                  f".{fn.attr}({v}) moves FUTURE values backward into "
                                  f"the current row. Use a positive shift to look back.")

            # Whole-series reductions computed once over all data.
            if fn.attr in {'max', 'min', 'mean', 'std', 'sum', 'median', 'quantile'}:
                if self._touches_array(fn.value):
                    self._add(
                        WARNING, 'whole-series-statistic', node,
                        f".{fn.attr}() over a full data array uses every bar including "
                        f"future ones. Compute it on a rolling window instead."
                    )
        self.generic_visit(node)

    @staticmethod
    def _negative_int(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return -node.operand.value
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value < 0:
            return node.value
        return None

    def _touches_array(self, node) -> bool:
        """Detects .array access, which exposes the entire series at once."""
        while isinstance(node, ast.Attribute):
            if node.attr == 'array':
                return True
            node = node.value
        return False

    # -- attribute reads: .array ------------------------------------------
    def visit_Attribute(self, node):
        if node.attr == 'array' and self._is_line_like(node.value):
            self._add(
                WARNING, 'raw-array-access', node,
                "Accessing .array exposes the entire series, including bars after "
                "the current one. Safe only if you slice strictly backwards."
            )
        self.generic_visit(node)


def _looks_like_bt_strategy(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                txt = ast.unparse(b) if hasattr(ast, 'unparse') else ''
                if 'Strategy' in txt:
                    return True
    return False


# ==============================================================================
# LAYER 2: EMPIRICAL PERTURBATION
# ==============================================================================

@dataclass
class PerturbationResult:
    clean: bool
    cut_points: List[int] = field(default_factory=list)
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    baseline_entries: int = 0
    targeted_cuts: int = 0        # cuts placed directly after a decision bar
    error: Optional[str] = None
    name: str = ''

    @property
    def power(self) -> str:
        """
        How much this run could have detected. A test that exercised no
        decisions proves nothing, and must not be read as a pass.
        """
        if self.error:
            return 'none'
        if self.targeted_cuts >= 12:
            return 'high'
        if self.targeted_cuts >= 5:
            return 'moderate'
        if self.targeted_cuts >= 1:
            return 'low'
        return 'none'

    def summary(self) -> str:
        lines = [f"\n{'=' * 68}", f"  PERTURBATION TEST: {self.name}", '=' * 68]
        if self.error:
            lines += [f"  [ERROR] {self.error}", '=' * 68]
            return '\n'.join(lines)
        lines.append(f"  Baseline entries:  {self.baseline_entries}")
        lines.append(f"  Cut points:        {len(self.cut_points)} "
                     f"({self.targeted_cuts} placed on decision bars)")
        lines.append(f"  Detection power:   {self.power}")
        if self.clean:
            lines.append("  No entry decision changed when future bars were replaced.")
            if self.power in ('low', 'none'):
                lines.append("")
                lines.append("  [WARN] Low power -- too few decisions were exercised for")
                lines.append("         this to be meaningful. Use more data or more cuts.")
                lines.append("  VERDICT: INCONCLUSIVE")
            else:
                lines.append("  VERDICT: PASS")
        else:
            lines.append(f"  {len(self.mismatches)} divergence(s) BEFORE the cut point:")
            for m in self.mismatches[:8]:
                lines.append(f"    cut={m['cut']}  {m['detail']}")
            lines.append("")
            lines.append("  An entry before the cut changed when only data AFTER it was")
            lines.append("  altered. That information can only have come from the future.")
            lines.append("  VERDICT: FAIL - lookahead confirmed")
        lines.append('=' * 68)
        return '\n'.join(lines)


def _make_entry_analyzer():
    """
    Analyzer recording every order the strategy submits, timestamped by the bar
    it was CREATED on. Non-invasive -- the strategy need not define or preserve
    notify_order.

    Why order.created.dt and not the current bar:
    Backtrader delivers the Submitted notification at the start of the NEXT bar
    cycle, so reading data.datetime.datetime(0) inside notify_order yields
    submission_bar + 1. That one-bar lag is quietly fatal here -- it pushes the
    final decision before a cut point past the cut, so exactly the decision the
    perturbation was designed to contaminate gets filtered out of the
    comparison, and a strategy reading close[1] is reported as clean.
    order.created.dt is the creation timestamp and carries no lag.
    """
    import backtrader as bt

    class EntryRecorder(bt.Analyzer):
        def start(self):
            self.entries = []

        def notify_order(self, order):
            if order.status != order.Submitted:
                return
            try:
                dt = bt.num2date(order.created.dt)
            except Exception:
                dt = None
            self.entries.append({
                'dt': dt,
                'isbuy': bool(order.isbuy()),
                'size': float(order.created.size or 0),
            })

        def get_analysis(self):  # pyright: ignore[reportIncompatibleMethodOverride]
            # Backtrader's Analyzer.get_analysis returns an AutoOrderedDict.
            # A plain list is what every caller here wants and what Backtrader
            # accepts, so the override is deliberate rather than an oversight.
            return self.entries

    return EntryRecorder


def perturb_future(df: pd.DataFrame, cut: int, seed: int = 0) -> pd.DataFrame:
    """
    Return a copy of df identical up to `cut`, with everything after replaced by
    a different but VALID price path.

    Validity matters: an invalid OHLC bar (low > high) could make a strategy
    behave oddly for reasons unrelated to lookahead, producing a false positive.
    The replacement is a random walk anchored at the last real close, with
    high/low derived so that low <= min(open, close) and high >= max(open, close).
    """
    out = df.copy()
    n = len(df) - cut
    if n <= 0:
        return out

    rng = np.random.RandomState(seed)
    anchor = float(df['close'].iloc[cut - 1]) if cut > 0 else float(df['close'].iloc[0])

    # Deliberately large moves: a weak perturbation may not change any decision
    # even for a strategy that genuinely peeks.
    steps = rng.normal(0.0, 0.02, n)
    path = anchor * np.exp(np.cumsum(steps))

    closes = path
    opens = np.concatenate([[anchor], closes[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    lows = np.maximum(lows, 1e-9)

    idx = out.index[cut:]
    out.loc[idx, 'open'] = opens
    out.loc[idx, 'high'] = highs
    out.loc[idx, 'low'] = lows
    out.loc[idx, 'close'] = closes
    if 'volume' in out.columns:
        out.loc[idx, 'volume'] = np.abs(rng.normal(1000, 300, n))
    return out


class LookaheadDetector:
    """Two-layer lookahead gate. See module docstring."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # -- LAYER 1 -----------------------------------------------------------
    def scan_source(self, source: str, name: str = '<string>') -> ScanReport:
        """Static AST scan. Fast, no data needed. Cannot prove absence."""
        report = ScanReport(name=name)
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            report.parse_error = f"line {e.lineno}: {e.msg}"
            return report

        v = _Visitor(source.splitlines())
        v.visit(tree)
        report.findings = v.findings

        if not _looks_like_bt_strategy(tree):
            report.findings.append(Finding(
                INFO, 'not-a-bt-strategy', 1, '',
                "No bt.Strategy subclass found; Backtrader-specific rules may not apply."
            ))
        return report

    def scan_file(self, path: str) -> ScanReport:
        with open(path, 'r', encoding='utf-8') as f:
            return self.scan_source(f.read(), name=os.path.basename(path))

    # -- LAYER 2 -----------------------------------------------------------
    def perturbation_test(
        self,
        strategy_class,
        data: pd.DataFrame,
        n_cuts: int = 20,
        strategy_params: Optional[Dict[str, Any]] = None,
        cash: float = 100_000,
        name: str = '',
        cut_fractions=None,
    ) -> PerturbationResult:
        """
        Run the strategy on real data and on futures-replaced data, and compare
        every entry decision made BEFORE the cut.

        WHERE THE CUTS GO -- this determines whether the test works at all.

        A strategy that peeks k bars ahead only contaminates the k decisions
        immediately before the cut; everything earlier never saw perturbed data.
        So a handful of arbitrary cut points is close to a coin flip: with three
        cuts and a one-bar peek, the detector misses roughly half the time.

        Cuts are therefore placed at (decision_bar + 1) for a spread of bars
        where the baseline actually submitted an order. Bar cut-1 is then known
        to be a bar the strategy acted on, so if that action consulted any
        future bar, the perturbation lands squarely on it. `targeted_cuts`
        records how many cuts were placed this way, and `power` reports whether
        the run had enough of them to mean anything.

        Only ENTRIES before the cut are compared. A position opened before the
        cut and closed after it legitimately exits differently, because the exit
        genuinely depends on post-cut prices -- comparing exits would
        manufacture false positives.

        Args:
            n_cuts: how many cut points to test. More cuts, more sensitivity,
                    proportionally more backtests.
            cut_fractions: override with explicit fractions of the series.
                    Bypasses decision targeting; mainly for reproducing a
                    specific case.
        """
        name = name or getattr(strategy_class, '__name__', 'strategy')
        result = PerturbationResult(clean=True, name=name)

        try:
            import backtrader as bt
        except ImportError:
            result.error = "backtrader not installed"
            result.clean = False
            return result

        required = {'open', 'high', 'low', 'close'}
        if not required.issubset(set(data.columns)):
            result.error = f"data needs columns {sorted(required)}, got {list(data.columns)}"
            result.clean = False
            return result
        if len(data) < 60:
            result.error = f"need at least 60 bars, got {len(data)}"
            result.clean = False
            return result

        Recorder = _make_entry_analyzer()

        def run(df):
            # Backtrader builds its kwargs through a metaclass, so a static
            # checker cannot see stdstats/dataname. Both are valid; the ignores
            # are scoped to these two calls so real call errors elsewhere still
            # get reported.
            cerebro = bt.Cerebro(stdstats=False)  # pyright: ignore[reportCallIssue]
            cerebro.broker.setcash(cash)
            feed = bt.feeds.PandasData(dataname=df)  # pyright: ignore[reportCallIssue]
            cerebro.adddata(feed)
            cerebro.addstrategy(strategy_class, **(strategy_params or {}))
            cerebro.addanalyzer(Recorder, _name='entries')
            buf = io.StringIO()
            with redirect_stdout(buf):        # strategies often print
                strats = cerebro.run()
            return strats[0].analyzers.entries.get_analysis()

        try:
            baseline = run(data)
        except Exception as e:
            result.error = f"baseline run failed: {type(e).__name__}: {e}"
            result.clean = False
            return result

        result.baseline_entries = len(baseline)

        # ---- choose cut points ------------------------------------------
        n = len(data)
        lo, hi = max(30, int(n * 0.10)), n - 2      # leave warmup and a tail

        if cut_fractions is not None:
            cuts = [int(n * f) for f in cut_fractions]
            targeted = 0
        else:
            # Map each order's creation timestamp back to a positional index.
            decision_bars = set()
            for e in baseline:
                if e['dt'] is None:
                    continue
                try:
                    d = data.index.get_loc(pd.Timestamp(e['dt']))
                except KeyError:
                    continue
                if isinstance(d, slice) or not isinstance(d, (int, np.integer)):
                    continue
                if lo <= d <= hi:
                    decision_bars.add(int(d))

            decision_bars = sorted(decision_bars)
            if decision_bars:
                if len(decision_bars) > n_cuts:
                    step = len(decision_bars) / n_cuts
                    decision_bars = [decision_bars[int(i * step)] for i in range(n_cuts)]
                # cut = d + 1 puts the perturbation on the very next bar, so a
                # one-bar peek from bar d lands inside the perturbed region.
                cuts = [d + 1 for d in decision_bars]
                targeted = len(cuts)
            else:
                # Strategy never traded in the usable range: fall back to an
                # even spread and report the resulting lack of power.
                cuts = [int(lo + (hi - lo) * i / max(n_cuts - 1, 1)) for i in range(n_cuts)]
                targeted = 0

        cuts = sorted({c for c in cuts if lo <= c <= hi})
        result.cut_points = cuts
        result.targeted_cuts = targeted

        # ---- compare -----------------------------------------------------
        for i, cut in enumerate(cuts):
            cut_dt = data.index[cut]
            try:
                perturbed = run(perturb_future(data, cut, seed=1000 + i))
            except Exception as e:
                result.mismatches.append({
                    'cut': cut,
                    'detail': f"perturbed run failed: {type(e).__name__}: {e}",
                })
                result.clean = False
                continue

            a = [e for e in baseline if e['dt'] is not None and e['dt'] < cut_dt]
            b = [e for e in perturbed if e['dt'] is not None and e['dt'] < cut_dt]

            if len(a) != len(b):
                result.clean = False
                result.mismatches.append({
                    'cut': cut,
                    'detail': f"entry COUNT before cut differs: {len(a)} vs {len(b)}",
                })
                continue

            for j, (x, y) in enumerate(zip(a, b)):
                if x['dt'] != y['dt'] or x['isbuy'] != y['isbuy']:
                    result.clean = False
                    result.mismatches.append({
                        'cut': cut,
                        'detail': (f"entry #{j} differs: "
                                   f"{x['dt']}/{'BUY' if x['isbuy'] else 'SELL'} vs "
                                   f"{y['dt']}/{'BUY' if y['isbuy'] else 'SELL'}"),
                    })
                    break

        return result

    # -- COMBINED GATE -----------------------------------------------------
    def gate(self, source: str, strategy_class=None, data=None, name='strategy') -> bool:
        """
        Returns True if the strategy may proceed to evaluation.

        Layer 1 always runs. Layer 2 runs only if a class and data are supplied
        AND layer 1 passed -- there is no point spending backtests on something
        already known to read the future.
        """
        report = self.scan_source(source, name=name)
        if self.verbose:
            print(report.summary())
        if report.failed:
            return False

        if strategy_class is not None and data is not None:
            res = self.perturbation_test(strategy_class, data, name=name)
            if self.verbose:
                print(res.summary())
            return res.clean
        return True