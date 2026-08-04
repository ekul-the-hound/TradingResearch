# ==============================================================================
# apply_integration_fixes_patch.py
# ==============================================================================
# Phase 0 follow-up. Clears the four test_integration.py failures.
#
# PROVENANCE -- which of these did the Phase 0 work cause?
# -------------------------------------------------------
# Checked by running each test against the ORIGINAL, unpatched
# canonical_result.py:
#
#   CR.12 Null Sharpe    FAILED on the original too   -> pre-existing
#   CR.14 Empty trades   PASSED on the original       -> caused by item 5
#   CR.15 Missing fields FAILED on the original too   -> pre-existing
#   PIPE.05 Canonical    unrelated to canonical_result -> pre-existing
#
# ------------------------------------------------------------------------------
# FIX 1 -- CR.14 (caused by the item 5 patch)
# ------------------------------------------------------------------------------
# test_cr_14_empty_trades passes trades=[], which is falsy, so _compute_arrays
# takes the no-trade-list branch. It used to fabricate a 100-element Gaussian
# series; it now correctly returns None. This is exactly the same pin as CR.05
# -- the item 5 patcher inverted CR.05 and missed CR.14. Inverting it now.
#
# ------------------------------------------------------------------------------
# FIX 2 -- CR.12 and CR.15 (pre-existing, stale since an earlier fix)
# ------------------------------------------------------------------------------
# Both assert `cr.sharpe_ratio == 0`. But canonical_result.py declares:
#
#     sharpe_ratio: Optional[float] = None   # None = unmeasured (distinct from 0.0)
#
# and from_backtest uses result.get("sharpe_ratio") specifically to preserve
# None. That distinction is deliberate and worth keeping: a strategy whose
# Sharpe could not be computed is not the same as one that scored exactly zero,
# and collapsing them would let unmeasurable strategies rank alongside flat
# ones. The tests were simply never updated when that change landed, so they
# now assert the opposite of the intended behaviour. Re-pinned to `is None`.
#
# ------------------------------------------------------------------------------
# FIX 3 -- PIPE.05 (pre-existing; step 6 of the pipeline is dead code)
# ------------------------------------------------------------------------------
# run_pipeline.step_6_diversify calls an API that does not exist -- twice:
#
#     df = DiversificationFilter(max_correlation=self.config.max_correlation)
#         -> __init__(self, lineage_tracker=None). max_correlation is a field
#            on DiversityConfig, not a constructor argument.
#
#     surviving_ids = df.filter(returns_dict)
#         -> there is no .filter(). The method is
#            run(strategies, returns_dict, trade_dates_dict, config) and it
#            returns a DiversificationResult, not a list of ids.
#
# The call is wrapped in `except ImportError`, which does not catch TypeError,
# so this raises straight out of the pipeline. Diversification has never run.
# Rewritten against the real API, with the except widened so a failure in this
# step degrades to "keep all candidates" instead of killing the run.
#
# USAGE
#   python apply_integration_fixes_patch.py --dry-run
#   python apply_integration_fixes_patch.py
#   python apply_integration_fixes_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

BACKUP_SUFFIX = '.intfix_bak'

PATCHES = [
    {
        'file': 'test_integration.py',
        'name': 'CR.14: empty trades must not fabricate returns (item 5 fallout)',
        'marker': 'INTEGRATION-FIX-CR14',
        'old': '''def test_cr_14_empty_trades():
    raw = {"strategy_name": "t", "trades": [], "bars_tested": 100,
           "total_return_pct": 5, "starting_value": 10000,
           "sharpe_ratio": 0.8, "max_drawdown_pct": 5, "total_trades": 0}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.returns is not None''',
        'new': '''def test_cr_14_empty_trades():
    # INTEGRATION-FIX-CR14
    # trades=[] is falsy, so _compute_arrays takes the no-trade-list branch.
    # That branch used to fabricate 100 Gaussian returns; it now reports None.
    # Same pin as CR.05 -- an empty trade list is not a return series.
    raw = {"strategy_name": "t", "trades": [], "bars_tested": 100,
           "total_return_pct": 5, "starting_value": 10000,
           "sharpe_ratio": 0.8, "max_drawdown_pct": 5, "total_trades": 0}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.returns is None, "empty trades must not fabricate returns"
    assert cr.returns_source == "none"
    assert not cr.has_real_returns
    # The equity curve still exists, holding just the starting value.
    assert cr.equity_curve is not None and len(cr.equity_curve) == 1''',
    },
    {
        'file': 'test_integration.py',
        'name': 'CR.12: unmeasured Sharpe is None, not 0 (stale pin)',
        'marker': 'INTEGRATION-FIX-CR12',
        'old': '''def test_cr_12_null_sharpe():
    raw = {"strategy_name": "test", "sharpe_ratio": None, "total_trades": 0}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.sharpe_ratio == 0''',
        'new': '''def test_cr_12_null_sharpe():
    # INTEGRATION-FIX-CR12
    # canonical_result deliberately preserves None to mean "unmeasured", which
    # is distinct from a measured 0.0. Collapsing the two would let a strategy
    # whose Sharpe could not be computed rank alongside a genuinely flat one.
    # This assertion predates that change and asserted the opposite.
    raw = {"strategy_name": "test", "sharpe_ratio": None, "total_trades": 0}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.sharpe_ratio is None''',
    },
    {
        'file': 'test_integration.py',
        'name': 'CR.15: missing fields leave Sharpe None (stale pin)',
        'marker': 'INTEGRATION-FIX-CR15',
        'old': '''def test_cr_15_missing_fields():
    raw = {"strategy_name": "minimal"}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.strategy_name == "minimal"
    assert cr.total_trades == 0
    assert cr.sharpe_ratio == 0''',
        'new': '''def test_cr_15_missing_fields():
    # INTEGRATION-FIX-CR15
    # A result dict with no sharpe_ratio key yields None (unmeasured), not 0.0.
    raw = {"strategy_name": "minimal"}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.strategy_name == "minimal"
    assert cr.total_trades == 0
    assert cr.sharpe_ratio is None''',
    },
    {
        'file': 'run_pipeline.py',
        'name': 'PIPE.05: call the DiversificationFilter API that actually exists',
        'marker': 'INTEGRATION-FIX-PIPE05',
        'old': '''        try:
            from diversification_filter import DiversificationFilter
            df = DiversificationFilter(max_correlation=self.config.max_correlation)
            returns_dict = {}
            for cr in candidates:
                if cr.returns is not None and len(cr.returns) > 10:
                    returns_dict[cr.strategy_id] = cr.returns

            if returns_dict:
                surviving_ids = df.filter(returns_dict)
                diversified = [cr for cr in candidates if cr.strategy_id in surviving_ids]
            else:
                diversified = candidates
        except ImportError:
            self._log("  [WARN]  diversification_filter not available")
            diversified = candidates''',
        'new': '''        try:
            # INTEGRATION-FIX-PIPE05
            # This block previously called two things that do not exist:
            #   DiversificationFilter(max_correlation=...)  -- the constructor
            #       takes lineage_tracker only; max_correlation is a field on
            #       DiversityConfig.
            #   df.filter(returns_dict)                     -- the method is
            #       run(strategies, returns_dict, trade_dates_dict, config)
            #       and it returns a DiversificationResult, not a list of ids.
            # Both raise TypeError/AttributeError, which `except ImportError`
            # does not catch, so this step has never completed. The except is
            # widened below so a failure here degrades to "keep everything"
            # rather than aborting the pipeline.
            from diversification_filter import DiversificationFilter, DiversityConfig

            returns_dict = {}
            for cr in candidates:
                if cr.returns is not None and len(cr.returns) > 10:
                    returns_dict[cr.strategy_id] = cr.returns

            if returns_dict:
                strategies = [
                    {
                        "strategy_id": cr.strategy_id,
                        "name": cr.strategy_name or cr.strategy_id,
                        "composite_score": cr.sharpe_ratio if cr.sharpe_ratio is not None else 0.0,
                    }
                    for cr in candidates
                ]
                dcfg = DiversityConfig(max_correlation=self.config.max_correlation)
                result = DiversificationFilter().run(
                    strategies, returns_dict=returns_dict, config=dcfg)
                surviving_ids = {
                    s.get("strategy_id", s.get("name")) for s in result.selected
                }
                diversified = [cr for cr in candidates if cr.strategy_id in surviving_ids]
                self._log(f"  [INFO] max pairwise corr {result.max_pairwise_corr:.2f}, "
                          f"effective N {result.effective_n:.1f}")
            else:
                self._log("  [WARN]  No candidate has a usable return series; "
                          "diversification skipped")
                diversified = candidates
        except ImportError:
            self._log("  [WARN]  diversification_filter not available")
            diversified = candidates
        except Exception as e:
            self._log(f"  [WARN]  Diversification failed ({type(e).__name__}: {e}); "
                      f"keeping all candidates")
            diversified = candidates''',
    },
]

POST_CONDITIONS = [
    ('test_integration.py', 'INTEGRATION-FIX-CR14', 'CR.14 not re-pinned'),
    ('test_integration.py', 'INTEGRATION-FIX-CR12', 'CR.12 not re-pinned'),
    ('test_integration.py', 'INTEGRATION-FIX-CR15', 'CR.15 not re-pinned'),
    ('run_pipeline.py', 'INTEGRATION-FIX-PIPE05', 'step_6_diversify not repaired'),
    ('run_pipeline.py', 'DiversityConfig(max_correlation', 'config not routed correctly'),
]

ABSENT_CONDITIONS = [
    ('run_pipeline.py', 'df.filter(returns_dict)', 'nonexistent .filter() still called'),
    ('run_pipeline.py', 'DiversificationFilter(max_correlation=',
     'constructor still called with an unsupported kwarg'),
]


def code_without_comments(path):
    """
    Source with comments stripped, for absent-checks.

    A naive substring search over the raw file matches the explanatory comments
    that document the bug being removed -- which quote the broken call verbatim
    and would therefore report the fix as having failed. ast.unparse drops
    comments while preserving code, so the check looks only at what runs.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return ast.unparse(ast.parse(f.read()))
    except Exception:
        # Fallback: drop whole-line comments only.
        txt, _ = read_text(path)
        return '\n'.join(l for l in txt.split('\n') if not l.strip().startswith('#'))


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
    for p in PATCHES:
        by_file.setdefault(p['file'], []).append(p)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backups, failed_any = {}, False

    for filename, patches in by_file.items():
        path = os.path.join(project_dir, filename)
        print(f"\n{'=' * 70}")
        print(f"FILE: {filename}")
        print('=' * 70)

        if not os.path.exists(path):
            print(f"  [FAIL] Not found: {path}")
            failed_any = True
            continue

        text, crlf = read_text(path)
        applied, skipped, failed = [], [], []

        for p in patches:
            if p['marker'] in text:
                skipped.append(p['name'])
                continue
            count = text.count(p['old'])
            if count == 0:
                failed.append((p['name'], 'anchor not found - file differs from the snapshot'))
                continue
            if count > 1:
                failed.append((p['name'], f'anchor matched {count} times - ambiguous, refusing'))
                continue
            text = text.replace(p['old'], p['new'], 1)
            applied.append(p['name'])

        for n in applied:
            print(f"  [APPLY] {n}")
        for n in skipped:
            print(f"  [SKIP]  {n} (already patched)")
        for n, why in failed:
            print(f"  [FAIL]  {n}\n          {why}")

        if failed:
            print(f"\n  Refusing to partially patch {filename}. File unchanged.")
            failed_any = True
            continue
        if not applied:
            print("  Nothing to write.")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
            continue

        backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
        shutil.copy2(path, backup)
        backups[filename] = backup
        print(f"  [BACKUP] {os.path.basename(backup)}")

        write_text(path, text, crlf)

        ok, err = verify_syntax(path)
        if ok:
            print("  [VERIFY] Syntax OK")
        else:
            print(f"  [VERIFY] SYNTAX ERROR - {err}")
            shutil.copy2(backup, path)
            print("  [ROLLBACK] Restored from backup")
            failed_any = True

    if dry_run or failed_any:
        return not failed_any

    problems = []
    for filename, needle, msg in POST_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if os.path.exists(path):
            txt, _ = read_text(path)
            if needle not in txt:
                problems.append(f"{filename}: {msg}")
    for filename, needle, msg in ABSENT_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if os.path.exists(path):
            # Comment-free, so the check cannot match the comments that
            # document the very call being removed.
            if needle in code_without_comments(path):
                problems.append(f"{filename}: {msg}")

    print(f"\n{'=' * 70}")
    if problems:
        print("  [VERIFY] POST-CONDITIONS FAILED:")
        for p in problems:
            print(f"           - {p}")
        for filename, backup in backups.items():
            shutil.copy2(backup, os.path.join(project_dir, filename))
            print(f"  [ROLLBACK] {filename}")
        return False

    print(f"  [VERIFY] Post-conditions OK ({len(POST_CONDITIONS)} checked)")
    return True


def revert(project_dir):
    print("\nREVERT")
    print("=" * 70)
    done = False
    for filename in sorted({p['file'] for p in PATCHES}):
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
    ap = argparse.ArgumentParser(description="Fix the four test_integration failures")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)

    print("=" * 70)
    print("INTEGRATION FIXES - PATCHER")
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
        print("  python test_integration.py     (expect 28 passed, 0 failed)")
        print("\nNOTE: PIPE.05 was not a test problem. Step 6 of the pipeline")
        print("called two methods that do not exist and raised past its own")
        print("except clause, so diversification has never actually run.")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
