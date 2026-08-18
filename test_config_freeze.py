# ==============================================================================
# test_config_freeze.py -- Tests for the challenge config freeze / tamper guard
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import os
import json
import tempfile
import unittest

from config_freeze import ConfigFreeze, compute_hash, VerifyResult


CODE = "def next(self):\n    if self.rsi < 30:\n        self.buy()\n"
PARAMS = {"period": 14, "threshold": 30, "mult": 2.0}


class FreezeTestBase(unittest.TestCase):
    def setUp(self):
        self.mp = tempfile.mktemp(suffix=".json")
        self.cf = ConfigFreeze(manifest_path=self.mp)

    def tearDown(self):
        for suffix in ("", ".tmp"):
            p = self.mp + suffix
            if os.path.exists(p):
                os.remove(p)

    def _freeze_one(self, sid="s1", code=CODE, params=PARAMS):
        self.cf.freeze([{"strategy_id": sid, "code": code, "params": params}])


class TestHashing(unittest.TestCase):
    def test_same_input_same_hash(self):
        self.assertEqual(compute_hash(CODE, PARAMS), compute_hash(CODE, PARAMS))

    def test_code_change_changes_hash(self):
        self.assertNotEqual(compute_hash(CODE, PARAMS),
                            compute_hash(CODE + "\n    self.sell()", PARAMS))

    def test_param_change_changes_hash(self):
        self.assertNotEqual(compute_hash(CODE, PARAMS),
                            compute_hash(CODE, {**PARAMS, "period": 15}))

    def test_param_order_irrelevant(self):
        a = compute_hash(CODE, {"a": 1, "b": 2})
        b = compute_hash(CODE, {"b": 2, "a": 1})
        self.assertEqual(a, b)

    def test_crlf_vs_lf_same_hash(self):
        # A line-ending conversion must NOT trip the guard.
        self.assertEqual(compute_hash(CODE, PARAMS),
                        compute_hash(CODE.replace("\n", "\r\n"), PARAMS))

    def test_trailing_whitespace_ignored(self):
        noisy = "def next(self):   \n    pass  \n"
        clean = "def next(self):\n    pass\n"
        self.assertEqual(compute_hash(noisy, {}), compute_hash(clean, {}))

    def test_trailing_blank_lines_ignored(self):
        self.assertEqual(compute_hash("x = 1\n", {}),
                        compute_hash("x = 1\n\n\n", {}))

    def test_params_as_json_string(self):
        self.assertEqual(compute_hash(CODE, PARAMS),
                        compute_hash(CODE, json.dumps(PARAMS)))

    def test_comment_change_does_change_hash(self):
        # Comments are meaningful during a challenge -> must alter the hash.
        self.assertNotEqual(compute_hash("x = 1  # v1\n", {}),
                            compute_hash("x = 1  # v2\n", {}))


class TestFreezeVerify(FreezeTestBase):
    def test_unchanged_verifies(self):
        self._freeze_one()
        r = self.cf.verify("s1", CODE, PARAMS)
        self.assertTrue(r.ok)

    def test_changed_code_blocks(self):
        self._freeze_one()
        r = self.cf.verify("s1", CODE + "\n    self.sell()", PARAMS)
        self.assertFalse(r.ok)
        self.assertIn("mismatch", r.reason)

    def test_changed_param_blocks(self):
        self._freeze_one()
        r = self.cf.verify("s1", CODE, {**PARAMS, "period": 21})
        self.assertFalse(r.ok)

    def test_hashes_reported(self):
        self._freeze_one()
        r = self.cf.verify("s1", CODE, {**PARAMS, "period": 21})
        self.assertIsNotNone(r.expected_hash)
        self.assertIsNotNone(r.actual_hash)
        self.assertNotEqual(r.expected_hash, r.actual_hash)


class TestRefuseByDefault(FreezeTestBase):
    def test_no_manifest_refuses(self):
        # Nothing frozen yet -> nothing may run.
        r = self.cf.verify("s1", CODE, PARAMS)
        self.assertFalse(r.ok)
        self.assertIn("no freeze manifest", r.reason)

    def test_unknown_strategy_refuses(self):
        self._freeze_one("s1")
        r = self.cf.verify("s2", CODE, PARAMS)
        self.assertFalse(r.ok)
        self.assertIn("not in the frozen manifest", r.reason)

    def test_corrupt_manifest_refuses(self):
        self._freeze_one()
        # Corrupt the manifest file.
        with open(self.mp, "w") as f:
            f.write("{ not valid json")
        r = self.cf.verify("s1", CODE, PARAMS)
        self.assertFalse(r.ok)  # unreadable -> refuse


class TestOverride(FreezeTestBase):
    def test_override_permits_mismatch(self):
        self._freeze_one()
        self.cf.override("s1", "deliberate retune after week 1")
        r = self.cf.verify("s1", CODE, {**PARAMS, "period": 21})
        self.assertTrue(r.ok)
        self.assertTrue(r.overridden)

    def test_override_requires_reason(self):
        self._freeze_one()
        with self.assertRaises(ValueError):
            self.cf.override("s1", "")

    def test_override_before_freeze_raises(self):
        with self.assertRaises(RuntimeError):
            self.cf.override("s1", "reason")

    def test_clear_override_restores_block(self):
        self._freeze_one()
        self.cf.override("s1", "temporary")
        self.cf.clear_override("s1")
        r = self.cf.verify("s1", CODE, {**PARAMS, "period": 21})
        self.assertFalse(r.ok)  # blocked again

    def test_override_does_not_affect_unchanged_verify(self):
        # Override present but code unchanged -> ok without overridden flag.
        self._freeze_one()
        self.cf.override("s1", "just in case")
        r = self.cf.verify("s1", CODE, PARAMS)
        self.assertTrue(r.ok)
        self.assertFalse(r.overridden)  # matched cleanly, override not needed


class TestFreezeManifest(FreezeTestBase):
    def test_freeze_multiple(self):
        self.cf.freeze([
            {"strategy_id": "a", "code": "x=1", "params": {}},
            {"strategy_id": "b", "code": "y=2", "params": {"p": 1}},
        ])
        m = self.cf.get_manifest()
        self.assertEqual(len(m["entries"]), 2)

    def test_freeze_requires_strategy_id(self):
        with self.assertRaises(ValueError):
            self.cf.freeze([{"code": "x=1"}])

    def test_freeze_accepts_generated_code_key(self):
        self.cf.freeze([{"strategy_id": "a", "generated_code": "x=1"}])
        r = self.cf.verify("a", "x=1", {})
        self.assertTrue(r.ok)

    def test_is_frozen(self):
        self.assertFalse(self.cf.is_frozen())
        self._freeze_one()
        self.assertTrue(self.cf.is_frozen())

    def test_verify_all(self):
        self.cf.freeze([
            {"strategy_id": "a", "code": "x=1", "params": {}},
            {"strategy_id": "b", "code": "y=2", "params": {}},
        ])
        results = self.cf.verify_all([
            {"strategy_id": "a", "code": "x=1", "params": {}},
            {"strategy_id": "b", "code": "y=2-CHANGED", "params": {}},
        ])
        by_id = {r.strategy_id: r.ok for r in results}
        self.assertTrue(by_id["a"])
        self.assertFalse(by_id["b"])


class TestPersistence(FreezeTestBase):
    def test_manifest_survives_reload(self):
        self._freeze_one()
        cf2 = ConfigFreeze(manifest_path=self.mp)
        r = cf2.verify("s1", CODE, PARAMS)
        self.assertTrue(r.ok)

    def test_override_persists(self):
        self._freeze_one()
        self.cf.override("s1", "persisted reason")
        cf2 = ConfigFreeze(manifest_path=self.mp)
        r = cf2.verify("s1", CODE, {**PARAMS, "period": 99})
        self.assertTrue(r.ok)
        self.assertTrue(r.overridden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
