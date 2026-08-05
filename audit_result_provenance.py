# ==============================================================================
# audit_result_provenance.py
# ==============================================================================
# Reports which stored backtest results can still be trusted.
#
# THE SITUATION
# -------------
# The Phase 0 timezone fix shifted every forex timestamp by +5 hours and the
# forex cache was rebuilt. Results computed before that ran against a
# daily-loss boundary drawn five hours from where it belongs. They are still in
# backtest_results, in the same table, in the same format, next to correct ones.
#
# This tool separates them. It uses two independent signals:
#
#   1. FINGERPRINT PRESENT. Results written after apply_fingerprint_patch.py
#      carry a data hash. Anything without one predates fingerprinting, which
#      means it also predates the timezone fix.
#
#   2. START TIMESTAMP. HistData weeks open at 17:00 in its own EST clock,
#      which becomes 22:00 after conversion to UTC. A forex result whose data
#      starts in the 16:00-19:00 band was computed on unconverted data.
#
# Signal 2 is what makes this more than a formality: it can positively identify
# stale rows rather than merely failing to vouch for them.
#
# NOTHING IS DELETED. The tool reports and, with --tag, marks. Deciding what to
# discard is your call, not a script's.
#
# USAGE
#   python audit_result_provenance.py
#   python audit_result_provenance.py --tag        # write a stale_reason column
#   python audit_result_provenance.py --db path/to/results.db
# ==============================================================================

import argparse
import os
import sqlite3
import sys

try:
    import config
    DEFAULT_DB = config.DATABASE_PATH
except Exception:
    DEFAULT_DB = os.path.join('results', 'backtest_results.db')

try:
    import data_fingerprint as dfp
except ImportError:
    dfp = None


VERDICT_OK = 'OK'
VERDICT_STALE = 'STALE'
VERDICT_UNKNOWN = 'UNKNOWN'


def classify(row):
    """Return (verdict, reason)."""
    has_fp = bool(row.get('data_fingerprint'))
    start = row.get('start_date')
    symbol = row.get('symbol') or ''

    pre_fix = None
    if dfp is not None:
        pre_fix = dfp.looks_pre_timezone_fix(start, symbol)

    if pre_fix is True:
        return VERDICT_STALE, (
            f"data starts {str(start)[:19]} -- an unconverted EST timestamp. "
            f"Computed on a 5h-shifted daily boundary.")
    if has_fp and pre_fix is False:
        return VERDICT_OK, "fingerprinted, timestamps consistent with UTC"
    if has_fp:
        return VERDICT_OK, "fingerprinted"
    if pre_fix is False:
        return VERDICT_UNKNOWN, (
            "no fingerprint, but timestamps look post-conversion. "
            "Probably fine; cannot be verified.")
    return VERDICT_UNKNOWN, (
        "no fingerprint and timestamps are inconclusive. Predates provenance "
        "tracking, so it also predates the timezone fix.")


def audit(db_path, tag=False):
    if not os.path.exists(db_path):
        print(f"[FAIL] Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cols = {r['name'] for r in conn.execute("PRAGMA table_info(backtest_results)")}
    has_fp_col = 'data_fingerprint' in cols

    select = "SELECT id, strategy_name, symbol, timeframe, start_date, end_date, " \
             "total_return_pct, sharpe_ratio"
    if has_fp_col:
        select += ", data_fingerprint"
    select += " FROM backtest_results ORDER BY id"

    try:
        rows = [dict(r) for r in conn.execute(select).fetchall()]
    except sqlite3.OperationalError as e:
        print(f"[FAIL] Could not read backtest_results: {e}")
        conn.close()
        return 1

    print("=" * 78)
    print("  RESULT PROVENANCE AUDIT")
    print("=" * 78)
    print(f"  Database: {db_path}")
    print(f"  Rows:     {len(rows)}")
    if not has_fp_col:
        print("  [NOTE] No data_fingerprint column -- apply_fingerprint_patch.py")
        print("         has not been run. Every row here predates provenance")
        print("         tracking, and therefore predates the timezone fix.")
    print("=" * 78)

    if not rows:
        print("  No results stored.")
        conn.close()
        return 0

    counts = {VERDICT_OK: 0, VERDICT_STALE: 0, VERDICT_UNKNOWN: 0}
    tagged = 0

    print(f"\n  {'id':>4}  {'verdict':8} {'symbol':10} {'tf':7} "
          f"{'start':20} {'ret%':>8}")
    print("  " + "-" * 74)

    for r in rows:
        verdict, reason = classify(r)
        counts[verdict] += 1
        ret = r.get('total_return_pct')
        ret_s = f"{ret:8.2f}" if isinstance(ret, (int, float)) else " " * 8
        print(f"  {r['id']:>4}  {verdict:8} {str(r.get('symbol') or ''):10} "
              f"{str(r.get('timeframe') or ''):7} {str(r.get('start_date'))[:19]:20} {ret_s}")
        if verdict != VERDICT_OK:
            print(f"        -> {reason}")

        if tag and verdict != VERDICT_OK:
            try:
                conn.execute(
                    "UPDATE backtest_results SET stale_reason = ? WHERE id = ?",
                    (f"{verdict}: {reason}", r['id']))
                tagged += 1
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE backtest_results ADD COLUMN stale_reason TEXT")
                conn.execute(
                    "UPDATE backtest_results SET stale_reason = ? WHERE id = ?",
                    (f"{verdict}: {reason}", r['id']))
                tagged += 1

    if tag:
        conn.commit()
    conn.close()

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  OK       {counts[VERDICT_OK]:4}  fingerprinted and consistent")
    print(f"  STALE    {counts[VERDICT_STALE]:4}  positively identified as pre-timezone-fix")
    print(f"  UNKNOWN  {counts[VERDICT_UNKNOWN]:4}  cannot be vouched for")
    if tag:
        print(f"\n  Tagged {tagged} row(s) with stale_reason. Nothing deleted.")
    if counts[VERDICT_STALE] or counts[VERDICT_UNKNOWN]:
        print("\n  These results were computed on data that has since been corrected.")
        print("  Do not use them as a baseline, and do not rank them against")
        print("  results produced after the fix. Re-run anything you still care")
        print("  about; the rest is archive.")
    print("=" * 78)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Audit which stored results predate the data fixes")
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--tag', action='store_true',
                    help='Write a stale_reason column. Never deletes.')
    args = ap.parse_args()
    return audit(args.db, tag=args.tag)


if __name__ == '__main__':
    sys.exit(main())
