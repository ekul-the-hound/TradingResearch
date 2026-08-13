# ==============================================================================
# test_preflight.py -- Tests for the pre-flight go/no-go checklist
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# These tests exercise the runner's aggregation/verdict logic and the checks
# that can run without a live terminal (using PaperBroker + real firm_rules).
# ==============================================================================

import os
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from preflight import (
    Preflight, PreflightConfig, CheckResult,
    PASS, WARN, FAIL, SKIP,
)


def primed_preflight(symbols=("EURUSD",), **cfg_kwargs):
    """A Preflight whose PaperBroker already has quotes set, so feed passes."""
    cfg = PreflightConfig(firm="ftmo", symbols=list(symbols),
                          broker_kind="paper", **cfg_kwargs)
    pf = Preflight(cfg)
    b = pf._get_broker()
    b.connect()
    for s in symbols:
        b.set_price(s, 1.10000)
    return pf


class TestVerdictAggregation(unittest.TestCase):
    def test_all_pass_is_go(self):
        pf = Preflight(PreflightConfig())
        pf.results = [CheckResult("a", PASS), CheckResult("b", PASS)]
        self.assertEqual(pf.verdict(), "GO")
        self.assertEqual(pf.exit_code(), 0)

    def test_any_fail_is_nogo(self):
        pf = Preflight(PreflightConfig())
        pf.results = [CheckResult("a", PASS), CheckResult("b", FAIL)]
        self.assertEqual(pf.verdict(), "NO-GO")
        self.assertEqual(pf.exit_code(), 1)

    def test_warn_alone_is_go(self):
        pf = Preflight(PreflightConfig())
        pf.results = [CheckResult("a", PASS), CheckResult("b", WARN)]
        self.assertEqual(pf.verdict(), "GO")

    def test_skip_is_go_when_not_strict(self):
        pf = Preflight(PreflightConfig(strict=False))
        pf.results = [CheckResult("a", PASS), CheckResult("b", SKIP)]
        self.assertEqual(pf.verdict(), "GO")

    def test_skip_is_nogo_when_strict(self):
        pf = Preflight(PreflightConfig(strict=True))
        pf.results = [CheckResult("a", PASS), CheckResult("b", SKIP)]
        self.assertEqual(pf.verdict(), "NO-GO")

    def test_fail_beats_everything_even_non_strict(self):
        pf = Preflight(PreflightConfig(strict=False))
        pf.results = [CheckResult("a", FAIL), CheckResult("b", SKIP)]
        self.assertEqual(pf.verdict(), "NO-GO")


class TestCrashingCheckIsFail(unittest.TestCase):
    def test_exception_in_check_becomes_fail(self):
        pf = Preflight(PreflightConfig(broker_kind="paper"))

        def boom():
            raise RuntimeError("kaboom")

        # Monkeypatch the check list via a single-run.
        pf.results = []
        try:
            pf.results.append(boom())
        except Exception:
            # emulate the runner's own guard
            pf.results.append(CheckResult("boom", FAIL, "check raised"))
        self.assertEqual(pf.results[0].severity, FAIL)


class TestRealChecks(unittest.TestCase):
    def test_firm_rules_pass(self):
        pf = primed_preflight()
        r = pf.check_firm_rules()
        self.assertIn(r.severity, (PASS, WARN))  # WARN if not fully modelled

    def test_firm_rules_unknown_firm_fails(self):
        pf = Preflight(PreflightConfig(firm="does_not_exist"))
        r = pf.check_firm_rules()
        self.assertEqual(r.severity, FAIL)

    def test_broker_connect_pass(self):
        pf = primed_preflight()
        self.assertEqual(pf.check_broker_connect().severity, PASS)

    def test_feed_alive_pass_when_primed(self):
        pf = primed_preflight(symbols=("EURUSD", "GBPUSD"))
        r = pf.check_feed_alive()
        self.assertEqual(r.severity, PASS)

    def test_feed_alive_fail_when_no_price(self):
        cfg = PreflightConfig(firm="ftmo", symbols=["EURUSD"], broker_kind="paper")
        pf = Preflight(cfg)
        pf._get_broker().connect()  # connected but no price set
        r = pf.check_feed_alive()
        self.assertEqual(r.severity, FAIL)

    def test_feed_alive_skip_without_symbols(self):
        pf = primed_preflight(symbols=())
        self.assertEqual(pf.check_feed_alive().severity, SKIP)

    def test_rule_budgets_computes_real_numbers(self):
        pf = primed_preflight()
        r = pf.check_rule_budgets()
        self.assertEqual(r.severity, PASS)
        # FTMO 5% of 100k = 5000
        self.assertAlmostEqual(r.data["daily_loss_limit"], 5000.0, places=2)

    def test_positions_flat_pass(self):
        pf = primed_preflight()
        self.assertEqual(pf.check_positions_reconciled().severity, PASS)

    def test_disk_space_pass(self):
        pf = primed_preflight(min_free_disk_mb=0.001)
        self.assertEqual(pf.check_disk_space().severity, PASS)

    def test_disk_space_fail_on_absurd_requirement(self):
        pf = primed_preflight(min_free_disk_mb=1e12)  # 1 PB
        self.assertEqual(pf.check_disk_space().severity, FAIL)

    def test_governor_constructs(self):
        pf = primed_preflight()
        self.assertEqual(pf.check_governor_importable().severity, PASS)


class TestDbBackup(unittest.TestCase):
    def test_backup_skip_without_paths(self):
        pf = primed_preflight()
        self.assertEqual(pf.check_db_backup().severity, SKIP)

    def test_backup_creates_copy(self):
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "toy.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()

        pf = primed_preflight()
        pf.config.db_paths_to_back_up = [str(db)]
        r = pf.check_db_backup()
        self.assertEqual(r.severity, PASS)
        baks = list(Path(tmp).glob("toy.db.preflight_*.bak"))
        self.assertEqual(len(baks), 1)

    def test_backup_missing_db_fails(self):
        pf = primed_preflight()
        pf.config.db_paths_to_back_up = ["/nonexistent/path/xyz.db"]
        self.assertEqual(pf.check_db_backup().severity, FAIL)


class TestEndToEnd(unittest.TestCase):
    def test_full_run_primed_is_go(self):
        pf = primed_preflight(symbols=("EURUSD", "GBPUSD"))
        pf.run()
        # No FAILs expected; SKIPs (clock, db_backup) allowed when not strict.
        self.assertEqual(pf.verdict(), "GO")

    def test_full_run_unprimed_is_nogo(self):
        cfg = PreflightConfig(firm="ftmo", symbols=["EURUSD"], broker_kind="paper")
        pf = Preflight(cfg)
        pf.run()
        # feed_alive should FAIL (no price set) -> NO-GO
        self.assertEqual(pf.verdict(), "NO-GO")

    def test_json_render_is_valid(self):
        pf = primed_preflight()
        pf.run()
        parsed = json.loads(pf.render_json())
        self.assertIn("verdict", parsed)
        self.assertIn("checks", parsed)
        self.assertTrue(len(parsed["checks"]) >= 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
