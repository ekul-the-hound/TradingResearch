# ==============================================================================
# phase0_setup.py -- Phase 0: Verify Locked Decisions + Discard Stale Backtests
# ==============================================================================
# One command to (1) confirm the locked Phase 0 decisions are actually in place
# in the config, and (2) safely discard the old pre-timezone-fix backtests.
#
# LOCKED DECISIONS THIS SCRIPT VERIFIES:
#   * Firm            = FTMO
#   * Max daily loss  = 5%      (0.05)
#   * Max total DD    = 10%     (0.10)
#   * Challenge target= 10%     (0.10)
#   * Verification    = 5%      (0.05)
#   * Holdout cutoff  = 20%     (0.20)
#   * Param stability = real sweeps (analyze_1d/analyze_2d present)
#
# DISCARD SAFETY:
#   Deleting rows is irreversible, so this NEVER deletes without first writing a
#   timestamped backup copy of the whole database. It reports how many rows it
#   will remove, backs up, deletes, and VACUUMs. Requires --discard to actually
#   delete; without it, the script only reports (dry run).
#
# USAGE:
#   python phase0_setup.py                 # verify settings + dry-run the discard
#   python phase0_setup.py --discard       # verify + actually discard stale rows
#   python phase0_setup.py --db path.db    # point at a specific results DB
# ==============================================================================

from __future__ import annotations

import os
import sys
import shutil
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))


# Expected locked values.
EXPECTED = {
    "firm_name": "FTMO",
    "max_daily_loss_pct": 0.05,
    "max_total_drawdown_pct": 0.10,
    "challenge_target": 0.10,
    "verification_target": 0.05,
    "holdout_fraction": 0.20,
}

OK = "[ OK ]"
BAD = "[FAIL]"
INFO = "[info]"


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ── Verification ──────────────────────────────────────────────────────────────
def verify_firm_rules() -> List[Tuple[bool, str]]:
    out: List[Tuple[bool, str]] = []
    try:
        from firm_rules import load_profile
        rules = load_profile("ftmo")
    except Exception as e:
        return [(False, f"could not load FTMO profile: {e}")]

    checks = [
        ("firm_name", getattr(rules, "firm_name", None), EXPECTED["firm_name"]),
        ("max_daily_loss_pct", getattr(rules, "max_daily_loss_pct", None),
         EXPECTED["max_daily_loss_pct"]),
        ("max_total_drawdown_pct", getattr(rules, "max_total_drawdown_pct", None),
         EXPECTED["max_total_drawdown_pct"]),
    ]
    for name, actual, expected in checks:
        if isinstance(expected, str):
            ok = str(actual) == expected
        else:
            ok = actual is not None and _close(actual, expected)
        out.append((ok, f"FTMO {name}: {actual} (expected {expected})"))

    # Profit targets live in a dict.
    targets = getattr(rules, "profit_targets", {}) or {}
    ch = targets.get("challenge")
    ve = targets.get("verification")
    out.append((ch is not None and _close(ch, EXPECTED["challenge_target"]),
                f"FTMO challenge target: {ch} (expected "
                f"{EXPECTED['challenge_target']})"))
    out.append((ve is not None and _close(ve, EXPECTED["verification_target"]),
                f"FTMO verification target: {ve} (expected "
                f"{EXPECTED['verification_target']})"))
    return out


def verify_holdout() -> List[Tuple[bool, str]]:
    try:
        import holdout_guard
        frac = getattr(holdout_guard, "DEFAULT_HOLDOUT_FRACTION", None)
    except Exception as e:
        return [(False, f"could not read holdout_guard: {e}")]
    ok = frac is not None and _close(frac, EXPECTED["holdout_fraction"])
    return [(ok, f"holdout fraction: {frac} (expected "
                 f"{EXPECTED['holdout_fraction']})")]


def verify_param_stability() -> List[Tuple[bool, str]]:
    try:
        import parameter_stability as ps
    except Exception as e:
        return [(False, f"could not import parameter_stability: {e}")]
    has_sweeps = hasattr(ps, "analyze_1d") and hasattr(ps, "analyze_2d")
    return [(has_sweeps,
             "parameter_stability real-sweep functions (analyze_1d/analyze_2d) "
             "present" if has_sweeps else
             "parameter_stability missing analyze_1d/analyze_2d")]


# ── Stale-backtest discard ────────────────────────────────────────────────────
def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone()
    return r is not None


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return col in cols


def count_stale(db_path: str) -> Tuple[int, int, str]:
    """
    Returns (stale_count, total_count, method). Prefers a stale_reason tag if
    audit_result_provenance --tag has been run; otherwise counts everything as
    the discard target (user chose to discard the whole old pool).
    """
    if not os.path.exists(db_path):
        return (0, 0, "no database file")
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        if not _table_exists(conn, "backtest_results"):
            return (0, 0, "no backtest_results table")
        total = conn.execute(
            "SELECT COUNT(*) FROM backtest_results").fetchone()[0]
        if _has_column(conn, "backtest_results", "stale_reason"):
            stale = conn.execute(
                "SELECT COUNT(*) FROM backtest_results "
                "WHERE stale_reason IS NOT NULL AND stale_reason != ''"
            ).fetchone()[0]
            return (stale, total, "stale_reason tag")
        # No tag column: the user's decision is to discard the whole old pool.
        return (total, total, "whole pool (no tag column; user chose discard)")
    finally:
        conn.close()


def backup_db(db_path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{db_path}.pre_discard_{stamp}.bak"
    shutil.copy2(db_path, dst)
    return dst


def discard_stale(db_path: str, method: str) -> int:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        if method == "stale_reason tag":
            cur = conn.execute(
                "DELETE FROM backtest_results "
                "WHERE stale_reason IS NOT NULL AND stale_reason != ''")
        else:
            cur = conn.execute("DELETE FROM backtest_results")
        deleted = cur.rowcount
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
        return deleted
    finally:
        conn.close()


# ── Orchestration ─────────────────────────────────────────────────────────────
def run(db_path: str, do_discard: bool) -> int:
    print("=" * 64)
    print(" PHASE 0 SETUP -- verify locked decisions + discard stale backtests")
    print("=" * 64)

    all_checks: List[Tuple[bool, str]] = []
    all_checks += verify_firm_rules()
    all_checks += verify_holdout()
    all_checks += verify_param_stability()

    print("\n-- Config verification --")
    fails = 0
    for ok, msg in all_checks:
        print(f"{OK if ok else BAD} {msg}")
        if not ok:
            fails += 1

    print("\n-- Stale backtest discard --")
    stale, total, method = count_stale(db_path)
    print(f"{INFO} db: {db_path}")
    print(f"{INFO} method: {method}")
    print(f"{INFO} {stale} of {total} row(s) targeted for discard")

    if stale == 0:
        print(f"{INFO} nothing to discard.")
    elif not do_discard:
        print(f"{INFO} DRY RUN -- pass --discard to actually delete "
              f"(a backup is written first).")
    else:
        backup = backup_db(db_path)
        print(f"{INFO} backup written: {backup}")
        deleted = discard_stale(db_path, method)
        print(f"{OK} discarded {deleted} row(s); database vacuumed.")

    print("\n" + "-" * 64)
    if fails:
        print(f" {fails} config check(s) FAILED -- fix before proceeding to "
              f"Phase 1.")
        return 1
    print(" All config checks passed. Phase 0 decisions are locked in.")
    if stale > 0 and not do_discard:
        print(" Re-run with --discard to clear the old backtests.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 0: verify locked decisions + discard stale backtests")
    # Default DB path: try config, else the conventional location.
    default_db = "data/backtest_results.db"
    # (results DB path is taken from --db; default below)
    ap.add_argument("--db", default=default_db,
                    help="Path to the results database")
    ap.add_argument("--discard", action="store_true",
                    help="Actually delete stale rows (writes a backup first)")
    args = ap.parse_args()
    return run(args.db, args.discard)


if __name__ == "__main__":
    raise SystemExit(main())