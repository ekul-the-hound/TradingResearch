# ==============================================================================
# apply_phase1_gates_patch.py
# ==============================================================================
# Phase 1 items 7 and 8 -- wiring, plus the source fix.
#
# ------------------------------------------------------------------------------
# THE SOURCE FIX (this is the important one)
# ------------------------------------------------------------------------------
# mutation_config.POSITION_SIZING is fed verbatim into the mutation prompt, and
# it currently instructs the LLM:
#
#     Martingale (increase after loss)
#     Scale in (multiple entries)
#     DCA (dollar cost averaging)
#
# Martingale is prohibited by essentially every prop firm. Scale-in and DCA
# describe averaging into a losing position, which is how grid strategies are
# built and is prohibited by most. So the pipeline is being told to generate
# strategies that cannot be funded no matter how well they backtest -- and then
# spending backtest time on them.
#
# A detector alone would catch these downstream, after the cost is paid. Fixing
# the prompt stops them being produced. Both are worth doing: the prompt is the
# source, the detector is the guarantee, and an LLM will occasionally produce a
# martingale without being asked.
#
# Anti-martingale and pyramiding are LEFT IN. Sizing up after WINS is a
# different thing and is generally permitted -- removing it would needlessly
# shrink the search space.
#
# ------------------------------------------------------------------------------
# WIRING
# ------------------------------------------------------------------------------
# run_pipeline's _lookahead_gate becomes _safety_gate, running all three static
# checks in one pass: lookahead, prohibited patterns, and crash resistance.
#
# Cost ordering matters. Static AST scans cost milliseconds; the crash suite
# costs ten short backtests. So source scans run first and a rejection
# short-circuits before any backtest is spent.
#
# The behavioural prohibited-pattern check is NOT in this gate -- it needs a
# trade list, which only exists after evaluation. It belongs at promotion, and
# is now possible because trades are persisted.
#
# USAGE
#   python apply_phase1_gates_patch.py --dry-run
#   python apply_phase1_gates_patch.py
#   python apply_phase1_gates_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

BACKUP_SUFFIX = '.gates_bak'

PATCHES = [
    {
        'file': 'mutation_config.py',
        'name': 'Stop instructing the LLM to generate prohibited patterns',
        'marker': 'PROHIBITED-PATTERN-SOURCE-FIX',
        'old': '''POSITION_SIZING = """
Fixed percentage risk per trade (1%, 2%)
Volatility-adjusted position size
Martingale (increase after loss)
Anti-martingale (increase after win)
Kelly criterion
Scale in (multiple entries)
DCA (dollar cost averaging)
Pyramid (add to winners)
"""''',
        'new': '''# PROHIBITED-PATTERN-SOURCE-FIX
# Removed: "Martingale (increase after loss)", "Scale in (multiple entries)",
# "DCA (dollar cost averaging)".
#
# This text goes straight into the mutation prompt, so it was instructing the
# LLM to produce strategies that prop firms prohibit outright. Such a strategy
# cannot be funded regardless of its backtest, so every hour spent evaluating
# one is wasted -- and the pipeline had no detector to catch them.
#
# Anti-martingale and pyramiding are KEPT. Sizing up after WINS is a different
# behaviour and is generally permitted; removing it would shrink the search
# space for no benefit.
#
# prohibited_patterns.py catches these downstream if an LLM produces one
# unprompted, which it occasionally will.
POSITION_SIZING = """
Fixed percentage risk per trade (0.25%, 0.5%, 1%)
Volatility-adjusted position size (ATR-scaled)
Anti-martingale (increase after win)
Kelly criterion (fractional, capped)
Pyramid (add to winners only, never to losers)
Fixed lot size
Equity-curve-based sizing (reduce after drawdown)

NEVER generate: martingale or any size increase after a loss; grid or
averaging down into a losing position; adding to a position that has moved
against you. Prop firms prohibit these and will void the account.
"""''',
    },
    {
        'file': 'run_pipeline.py',
        'name': 'Extend the lookahead gate into a combined safety gate',
        'marker': 'SAFETY-GATE',
        'old': '''            report = det.scan_file(src_path)
            if report.failed:
                rules = ', '.join(sorted({f.rule for f in report.critical})) or 'parse error'
                rejected.append((cr, rules))
                if hasattr(cr, 'strategy_params'):
                    cr.strategy_params['lookahead_rejected'] = rules
                self._log(f"  [REJECT] {cr.strategy_id}: lookahead ({rules})")
            else:
                kept.append(cr)''',
        'new': '''            # SAFETY-GATE
            # Three static checks in one pass, cheapest first so a rejection
            # short-circuits before any backtest is spent.
            reasons = []

            report = det.scan_file(src_path)
            if report.failed:
                rules = ', '.join(sorted({f.rule for f in report.critical})) or 'parse error'
                reasons.append(f"lookahead({rules})")

            # Prohibited patterns: martingale, grid, hedging. A strategy using
            # these cannot be funded regardless of performance, so this is a
            # harder gate than any metric threshold. Note the mutation prompt
            # used to actively request martingale -- see mutation_config.
            if not reasons:
                try:
                    import prohibited_patterns as _pp
                    prep = _pp.scan_file(src_path)
                    if prep.failed:
                        pats = ', '.join(sorted(prep.patterns))
                        reasons.append(f"prohibited({pats})")
                except ImportError:
                    self._log("  [WARN]  prohibited_patterns not available")

            if reasons:
                why = ' '.join(reasons)
                rejected.append((cr, why))
                if hasattr(cr, 'strategy_params'):
                    cr.strategy_params['safety_gate_rejected'] = why
                self._log(f"  [REJECT] {cr.strategy_id}: {why}")
            else:
                kept.append(cr)''',
    },
]

POST_CONDITIONS = [
    ('mutation_config.py', 'PROHIBITED-PATTERN-SOURCE-FIX', 'mutation prompt not fixed'),
    ('mutation_config.py', 'NEVER generate', 'explicit prohibition not added'),
    ('run_pipeline.py', 'SAFETY-GATE', 'gate not extended'),
    ('run_pipeline.py', 'import prohibited_patterns', 'pattern check not wired'),
]

ABSENT_CONDITIONS = [
    ('mutation_config.py', 'Martingale (increase after loss)',
     'the mutation prompt still requests martingale'),
    ('mutation_config.py', 'DCA (dollar cost averaging)',
     'the mutation prompt still requests DCA'),
]


def read_text(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    return raw.replace('\r\n', '\n'), ('\r\n' in raw)


def write_text(path, text, crlf):
    out = text.replace('\n', '\r\n') if crlf else text
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)


def code_without_comments(path):
    """
    Comment-free source, for absent-checks only.

    Presence markers live in comments and must be checked against RAW text;
    absent-checks assert code was removed and must ignore the comments that
    document the removal, which quote it verbatim. Opposite treatment -- using
    one rule for both fails whichever way you pick it.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return ast.unparse(ast.parse(f.read()))
    except Exception:
        txt, _ = read_text(path)
        return '\n'.join(l for l in txt.split('\n') if not l.strip().startswith('#'))


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
        print(f"\n{'=' * 70}\nFILE: {filename}\n{'=' * 70}")

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
            print("  [ROLLBACK] Restored")
            failed_any = True

    if dry_run or failed_any:
        return not failed_any

    problems = []
    for filename, needle, msg in POST_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if os.path.exists(path):
            raw, _ = read_text(path)
            if needle not in raw:
                problems.append(f"{filename}: {msg}")
    for filename, needle, msg in ABSENT_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if os.path.exists(path):
            if needle in code_without_comments(path):
                problems.append(f"{filename}: {msg}")

    print(f"\n{'=' * 70}")
    if problems:
        print("  [VERIFY] POST-CONDITIONS FAILED:")
        for p in problems:
            print(f"           - {p}")
        for f, b in backups.items():
            shutil.copy2(b, os.path.join(project_dir, f))
            print(f"  [ROLLBACK] {f}")
        return False

    print(f"  [VERIFY] Post-conditions OK "
          f"({len(POST_CONDITIONS) + len(ABSENT_CONDITIONS)} checked)")
    return True


def revert(project_dir):
    print("\nREVERT\n" + "=" * 70)
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
    ap = argparse.ArgumentParser(description="Wire Phase 1 gates and fix the mutation prompt")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)
    print("=" * 70)
    print("PHASE 1 GATES - PATCHER")
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
        print("  python test_phase1_gates.py")
        print("  python test_lookahead_detector.py")
        print("  python test_system.py")
        print("\nAny variant already generated from the old prompt may contain a")
        print("martingale. Scan them before spending more time on them:")
        print("  python -c \"import prohibited_patterns as p, glob;\\")
        print("             [print(p.scan_file(f).summary()) for f in glob.glob('variant_*.py')]\"")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
