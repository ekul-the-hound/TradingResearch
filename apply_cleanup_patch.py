# ==============================================================================
# apply_cleanup_patch.py
# ==============================================================================
# Batch of independently-fixable items from the session report. Each patch is
# self-contained; a failure in one does not block the others within its file.
#
# 1. ftmo_compliance.simulate_pass_rate  -> real bootstrap        (item 11.3)
# 2. react_dashboard2 regime chart       -> unavailable state     (item 11.11a)
# 3. react_dashboard2 cost estimate      -> real cost model       (item 11.11b)
# 4. run_pipeline step 2                 -> lookahead gate wired  (item 11.16)
# 5. data_manager cache read             -> ISO8601 date parsing  (item 11.13)
# 6. manual_gates                        -> reset_session()       (item 11.14)
#
# ------------------------------------------------------------------------------
# 1. simulate_pass_rate -- the shuffle was a no-op
# ------------------------------------------------------------------------------
#     shuffled = trades.sample(frac=1, random_state=random_seed + i)
#     result   = self.validate(shuffled, ...)
#
# Each trade row carries its own entry_date/exit_date, and _build_equity_curve
# sorts events by timestamp before doing anything. Reordering rows changes
# NOTHING -- verified empirically: original and shuffled produce byte-identical
# return and drawdown. So 1,000 simulations produce 1,000 identical results,
# pass_rate is only ever exactly 0.0 or 1.0, and every reported percentile
# collapses to the same number.
#
# Delegates to pass_rate_simulator.py, which samples with replacement and
# re-dates onto a fresh calendar so both composition and path actually vary.
#
# ------------------------------------------------------------------------------
# 2 & 3. The last two dashboard fabrications
# ------------------------------------------------------------------------------
#     regime_rets = [8.2, -2.1, 1.5, 3.4, -5.8, 6.1]
# Six hardcoded numbers rendered under the title "Portfolio Performance by
# Regime". No regime performance is persisted anywhere in the schema, so there
# is nothing to compute this from -- the honest output is an unavailable state.
#
#     est_costs = [abs(v["ret"]) * 0.3 for v in vs[:8]]   # estimated
# An invented 30%-of-return haircut labelled "Net (est.)", while
# cost_adjusted_scoring.py sits unused. Rewired to the real cost model, with an
# unavailable state when it cannot run.
#
# Same class as the FTMO proxy badges fixed in Phase 0 item 3.
#
# ------------------------------------------------------------------------------
# 4. Lookahead gate
# ------------------------------------------------------------------------------
# lookahead_detector.py exists and passes 29 tests, but nothing calls it. A gate
# that runs after evaluation is pointless; it belongs in front of step 2, where
# candidates first cost real backtest time.
#
# ------------------------------------------------------------------------------
# 5 & 6. Small fixes
# ------------------------------------------------------------------------------
# data_manager parsed cached timestamps without a format, so pandas fell back to
# dateutil and parsed millions of rows one at a time.
# manual_gates had no reset_session(), reported by test_system test 11.
#
# USAGE
#   python apply_cleanup_patch.py --dry-run
#   python apply_cleanup_patch.py
#   python apply_cleanup_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

BACKUP_SUFFIX = '.cleanup_bak'

PATCHES = [
    # -------------------------------------------------------------------------
    {
        'file': 'ftmo_compliance.py',
        'name': 'simulate_pass_rate: real bootstrap (shuffle was a no-op)',
        'marker': 'PASS-RATE-FIX',
        'old': '''            # Shuffle trade order
            shuffled = trades.sample(frac=1, random_state=random_seed + i).reset_index(drop=True)
            
            # Validate
            result = self.validate(shuffled, account_size=account_size, phase=phase)''',
        'new': '''            # PASS-RATE-FIX
            # Was: trades.sample(frac=1, ...) then validate(shuffled).
            #
            # That shuffle is a NO-OP. Every trade row carries its own
            # entry_date and exit_date, and _build_equity_curve sorts events by
            # timestamp before doing anything, so reordering rows cannot change
            # the result. Verified: original and shuffled give byte-identical
            # return and drawdown. All 1,000 simulations were the same
            # simulation, pass_rate could only ever be 0.0 or 1.0, and the
            # percentiles below were all the same number.
            #
            # pass_rate_simulator samples WITH REPLACEMENT and re-dates onto a
            # fresh calendar, so composition and path both vary. Preferred over
            # this loop entirely -- see simulate_pass_rate_bootstrap().
            import pass_rate_simulator as _prs
            shuffled = _prs.build_synthetic_window(
                trades, window_days=30,
                rng=np.random.RandomState(random_seed + i))
            if shuffled.empty:
                continue

            result = self.validate(shuffled, account_size=account_size, phase=phase)''',
    },
    {
        'file': 'ftmo_compliance.py',
        'name': 'Add simulate_pass_rate_bootstrap() entry point',
        'marker': 'def simulate_pass_rate_bootstrap',
        'old': '''    def simulate_pass_rate(
        self,
        trades_df: pd.DataFrame,''',
        'new': '''    def simulate_pass_rate_bootstrap(
        self,
        trades_df: pd.DataFrame,
        account_size: int = 100_000,
        phase: str = 'challenge',
        n_simulations: int = 1000,
        window_days: int = 30,
        mode: str = 'block',
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Preferred pass-rate estimate.

        Samples trades with replacement over a fixed challenge window and lays
        them on a fresh calendar, so both the trade set and its path through
        time vary between simulations. Defaults to a stationary block bootstrap,
        which preserves streaks -- and streaks are what breach a daily-loss rule.

        Returns a PassRateResult; call .to_dict() for the legacy dict shape.
        Reports degenerate=True if every simulation somehow came out identical,
        which is the failure the old implementation had and never surfaced.
        """
        import pass_rate_simulator as _prs
        return _prs.simulate_pass_rate(
            self, trades_df, account_size=account_size, phase=phase,
            n_simulations=n_simulations, window_days=window_days,
            mode=mode, random_seed=random_seed, verbose=verbose,
        )

    def simulate_pass_rate(
        self,
        trades_df: pd.DataFrame,''',
    },
    # -------------------------------------------------------------------------
    {
        'file': 'react_dashboard2.py',
        'name': 'Regime chart: unavailable instead of hardcoded numbers',
        'marker': 'DASHBOARD-HONESTY-REGIME',
        'old': '''    regime_rets=[8.2,-2.1,1.5,3.4,-5.8,6.1]
    f3.add_trace(go.Bar(x=regimes,y=regime_rets,marker_color=[T["green"],T["red"],T["blue"],T["amber"],T["red"],T["green"]]))
    f3.update_layout(title="Portfolio Performance by Regime",yaxis_title="Return %")''',
        'new': '''    # DASHBOARD-HONESTY-REGIME
    # Was: regime_rets=[8.2,-2.1,1.5,3.4,-5.8,6.1] -- six hardcoded numbers
    # rendered as "Portfolio Performance by Regime". No per-regime performance
    # is persisted anywhere in the schema, so there is nothing to compute this
    # from. Same class of bug as the FTMO proxy badges: a plausible-looking
    # chart standing in for an answer the system does not have.
    f3.add_annotation(
        text=("Per-regime performance is not recorded.<br>"
              "Run backtests with regime analysis enabled to populate this."),
        showarrow=False, font=dict(size=13, color=T["dim"]),
        xref="paper", yref="paper", x=0.5, y=0.5)
    f3.update_layout(title="Portfolio Performance by Regime (no data)",
                     yaxis_title="Return %",
                     xaxis=dict(visible=False), yaxis=dict(visible=False))''',
    },
    {
        'file': 'react_dashboard2.py',
        'name': 'Cost chart: real cost model instead of a 30% guess',
        'marker': 'DASHBOARD-HONESTY-COSTS',
        'old': '''        est_costs=[abs(v["ret"])*0.3 for v in vs[:8]]  # estimated
        net_rets=[r-c for r,c in zip(raw_rets,est_costs)]''',
        'new': '''        # DASHBOARD-HONESTY-COSTS
        # Was: est_costs=[abs(v["ret"])*0.3 ...] -- an invented 30%-of-return
        # haircut labelled "Net (est.)", while cost_adjusted_scoring.py sat
        # unused. A made-up number presented beside a real one is worse than no
        # number, because the chart implies they are comparable.
        net_rets = []
        _cost_ok = True
        try:
            from cost_adjusted_scoring import CostAdjustedScorer
            _scorer = CostAdjustedScorer()
            for v in vs[:8]:
                adj = _scorer.adjust_result({
                    "symbol": v.get("symbol", "EUR-USD"),
                    "total_return_pct": v["ret"],
                    "total_trades": v.get("trades", 0),
                    "bars_tested": v.get("bars", 0),
                })
                net = adj.get("net_return_pct") if isinstance(adj, dict) else getattr(adj, "net_return_pct", None)
                if net is None:
                    _cost_ok = False
                    break
                net_rets.append(net)
        except Exception as _e:
            _cost_ok = False
            print(f"[WARN] Cost model unavailable: {_e}")

        if not _cost_ok:
            net_rets = []''',
    },
    {
        'file': 'react_dashboard2.py',
        'name': 'Cost chart: only draw the net series when it is real',
        'marker': 'DASHBOARD-HONESTY-NETBAR',
        'old': '''        f1.add_trace(go.Bar(name="Net (est.)",x=names,y=net_rets,marker_color=T["amber"]))''',
        'new': '''        # DASHBOARD-HONESTY-NETBAR: omit the series rather than plot a guess.
        if net_rets:
            f1.add_trace(go.Bar(name="Net (after costs)",x=names,y=net_rets,marker_color=T["amber"]))''',
    },
    # -------------------------------------------------------------------------
    {
        'file': 'data_manager.py',
        'name': 'Cache read: explicit ISO8601 parsing (was dateutil fallback)',
        'marker': 'CACHE-DATEFMT',
        'old': '''        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df''',
        'new': '''        if os.path.exists(cache_file):
            try:
                # CACHE-DATEFMT
                # Without an explicit format pandas cannot infer one and falls
                # back to dateutil, which parses timestamps one at a time --
                # painful across millions of cached rows. Caches are written by
                # to_csv() so they are always ISO8601.
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True,
                                     date_format='ISO8601')
                except (TypeError, ValueError):
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df''',
    },
    # -------------------------------------------------------------------------
    {
        'file': 'manual_gates.py',
        'name': 'ValidationGate.reset_session()',
        'marker': 'def reset_session',
        'old': '''    def get_session_summary(self) -> dict:''',
        'new': '''    def reset_session(self):
        """
        Clear session counters and start a fresh session.

        Reported missing by test_system test 11. Useful between pipeline runs
        in one process, so approved-cost totals do not accumulate across
        unrelated runs and trip a budget ceiling that was never really hit.
        """
        self.session_start = datetime.now()
        self.total_approved_cost = 0.0
        self.total_blocked = 0
        self.total_approved = 0
        self.decisions = []
        return self.get_session_summary()

    def get_session_summary(self) -> dict:''',
    },
]

# run_pipeline gate: separate because the anchor is long.
GATE_PATCH = {
    'file': 'run_pipeline.py',
    'name': 'Wire the lookahead detector in front of step 2',
    'marker': 'LOOKAHEAD-GATE',
    'old': '''    def step_2_backtest_filter(self):''',
    'new': '''    def _lookahead_gate(self, candidates):
        """
        LOOKAHEAD-GATE

        Static lookahead scan in front of evaluation. A strategy that reads
        future bars backtests beautifully and loses money live, and it inflates
        exactly the metrics used for promotion -- so the scan belongs BEFORE
        backtest time is spent, not after.

        Layer 1 only (AST, milliseconds, no data). The empirical perturbation
        test costs a backtest per cut point and belongs at promotion, not here.

        Fails open: if the detector is missing the pipeline continues rather
        than halting, but says so. Silent degradation is how the pipeline's
        dead steps stayed hidden for so long.
        """
        try:
            from lookahead_detector import LookaheadDetector
        except ImportError:
            self._log("  [WARN]  lookahead_detector not available - gate SKIPPED")
            return candidates, []

        det = LookaheadDetector()
        kept, rejected = [], []

        for cr in candidates:
            src_path = None
            for attr in ('source_path', 'file_path', 'strategy_path'):
                if getattr(cr, attr, None):
                    src_path = getattr(cr, attr)
                    break
            if not src_path:
                params = getattr(cr, 'strategy_params', {}) or {}
                src_path = params.get('source_path') or params.get('file_path')

            if not src_path or not os.path.exists(src_path):
                kept.append(cr)          # nothing to scan; not evidence of guilt
                continue

            report = det.scan_file(src_path)
            if report.failed:
                rules = ', '.join(sorted({f.rule for f in report.critical})) or 'parse error'
                rejected.append((cr, rules))
                if hasattr(cr, 'strategy_params'):
                    cr.strategy_params['lookahead_rejected'] = rules
                self._log(f"  [REJECT] {cr.strategy_id}: lookahead ({rules})")
            else:
                kept.append(cr)

        if rejected:
            self._log(f"  [GATE] Lookahead: {len(rejected)} rejected, {len(kept)} passed")
        else:
            self._log(f"  [GATE] Lookahead: all {len(kept)} candidates clean")
        return kept, rejected

    def step_2_backtest_filter(self):''',
}


def read_text(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    return raw.replace('\r\n', '\n'), ('\r\n' in raw)


def write_text(path, text, crlf):
    out = text.replace('\n', '\r\n') if crlf else text
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)


def verify_syntax(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def apply_patches(project_dir, dry_run=False):
    by_file = {}
    for p in PATCHES + [GATE_PATCH]:
        by_file.setdefault(p['file'], []).append(p)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    any_applied, failed_any = False, False

    for filename, patches in by_file.items():
        path = os.path.join(project_dir, filename)
        print(f"\n{'=' * 70}\nFILE: {filename}\n{'=' * 70}")

        if not os.path.exists(path):
            print("  [SKIP] Not present in this project")
            continue

        text, crlf = read_text(path)
        applied, skipped, failed = [], [], []

        for p in patches:
            if p['marker'] in text:
                skipped.append(p['name'])
                continue
            c = text.count(p['old'])
            if c == 0:
                failed.append((p['name'], 'anchor not found - file differs from the snapshot'))
                continue
            if c > 1:
                failed.append((p['name'], f'anchor matched {c} times - ambiguous, refusing'))
                continue
            text = text.replace(p['old'], p['new'], 1)
            applied.append(p['name'])

        for n in applied:
            print(f"  [APPLY] {n}")
        for n in skipped:
            print(f"  [SKIP]  {n} (already patched)")
        for n, w in failed:
            print(f"  [FAIL]  {n}\n          {w}")

        if failed:
            print(f"\n  Refusing to partially patch {filename}. File unchanged.")
            failed_any = True
            continue
        if not applied:
            print("  Nothing to write.")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
            any_applied = True
            continue

        backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
        shutil.copy2(path, backup)
        print(f"  [BACKUP] {os.path.basename(backup)}")
        write_text(path, text, crlf)

        ok, err = verify_syntax(path)
        if ok:
            print("  [VERIFY] Syntax OK")
            any_applied = True
        else:
            print(f"  [VERIFY] SYNTAX ERROR - {err}")
            shutil.copy2(backup, path)
            print("  [ROLLBACK] Restored")
            failed_any = True

    return not failed_any


def revert(project_dir):
    print("\nREVERT\n" + "=" * 70)
    done = False
    for filename in sorted({p['file'] for p in PATCHES + [GATE_PATCH]}):
        path = os.path.join(project_dir, filename)
        bks = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
        if not bks:
            print(f"  [SKIP] No backup for {filename}")
            continue
        shutil.copy2(bks[-1], path)
        print(f"  [OK] {filename}  <-  {os.path.basename(bks[-1])}")
        done = True
    if not done:
        print("\n  Nothing to revert.")
    return done


def main():
    ap = argparse.ArgumentParser(description="Batch cleanup of independently-fixable items")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)
    print("=" * 70)
    print("CLEANUP BATCH - PATCHER")
    print("=" * 70)
    print(f"Project: {project_dir}")
    if args.dry_run:
        print("Mode:    DRY RUN (no files will be modified)")

    if args.revert:
        revert(project_dir)
        return 0

    ok = apply_patches(project_dir, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print("DRY RUN COMPLETE - re-run without --dry-run to apply")
    elif ok:
        print("PATCH COMPLETE")
        print("=" * 70)
        print("\nNEXT:")
        print("  python test_pass_rate_simulator.py")
        print("  python test_system.py")
        print("\nEXPECT pass rates to MOVE, probably downward. The old number came")
        print("from 1,000 identical simulations; real resampling produces daily-loss")
        print("breaches the no-op never could.")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
