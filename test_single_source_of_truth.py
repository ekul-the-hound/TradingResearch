# ==============================================================================
# test_single_source_of_truth.py
# ==============================================================================
# Prop-firm thresholds have accumulated in four places over this project:
# module constants in ftmo_compliance, a FirmRules profile, KillSwitchConfig,
# and the governor. Each copy is fine on the day it is written and wrong the
# first time a limit is edited somewhere else.
#
# These tests assert that a single FirmRules profile reaches every component.
# They are deliberately structural: they read the source and check behaviour,
# rather than restating the numbers, because a test that hardcodes 5.0 becomes
# a fifth copy of the problem.
# ==============================================================================

import inspect
import re
import sys
import unittest

import firm_rules
from firm_rules import Capability, FirmRules, ftmo

TIGHT = FirmRules(
    firm_name='TightFirm',
    max_daily_loss_pct=0.03,
    max_total_drawdown_pct=0.06,
    min_trading_days=7,
    profit_targets={'challenge': 0.08, 'verification': 0.04},
    account_sizes=[25_000, 150_000],
)


# ==============================================================================
# THE CHECKER
# ==============================================================================

class TestCheckerReadsTheProfile(unittest.TestCase):

    def setUp(self):
        from ftmo_compliance import FTMOComplianceChecker
        self.checker_cls = FTMOComplianceChecker

    def test_profile_reaches_the_instance(self):
        ck = self.checker_cls(rules=TIGHT)
        self.assertAlmostEqual(ck.rules.max_daily_loss_pct, 0.03)
        self.assertEqual(ck.rules.firm_name, 'TightFirm')

    def test_default_is_still_ftmo(self):
        """Existing callers that pass nothing must be unaffected."""
        ck = self.checker_cls()
        self.assertAlmostEqual(ck.rules.max_daily_loss_pct, 0.05)
        self.assertAlmostEqual(ck.rules.max_total_drawdown_pct, 0.10)

    def test_account_sizes_come_from_the_profile(self):
        ck = self.checker_cls(rules=TIGHT)
        with self.assertRaises(ValueError) as ctx:
            ck.validate(_empty_trades(), account_size=100_000)
        self.assertIn('TightFirm', str(ctx.exception))

    def test_no_method_reads_the_module_constants(self):
        """
        THE STRUCTURAL GUARD.

        validate() was converted to self.rules while validate_intrabar() and
        generate_report() kept reading the constants, so a custom firm got two
        verdicts under different limits. Scanning the source catches the next
        method that forgets, which behavioural tests only catch if someone
        remembers to write one.
        """
        import ftmo_compliance
        src = inspect.getsource(ftmo_compliance)

        # Drop the definitions themselves and the compatibility aliases.
        body = re.sub(r'^\s*(MAX_DAILY_LOSS_PCT|MAX_TOTAL_DRAWDOWN_PCT|'
                      r'MIN_TRADING_DAYS|PROFIT_TARGETS|ACCOUNT_SIZES)\s*=.*$',
                      '', src, flags=re.M)
        # Comments may legitimately mention them.
        body = re.sub(r'#.*$', '', body, flags=re.M)

        offenders = []
        for const in ('MAX_DAILY_LOSS_PCT', 'MAX_TOTAL_DRAWDOWN_PCT',
                      'MIN_TRADING_DAYS', 'PROFIT_TARGETS', 'ACCOUNT_SIZES'):
            for line in body.splitlines():
                if const in line and 'self.rules' not in line:
                    offenders.append(line.strip())

        self.assertEqual(
            offenders, [],
            'These read a module constant instead of self.rules, so they '
            'ignore the configured firm:\n  ' + '\n  '.join(offenders))


def _empty_trades():
    import pandas as pd
    return pd.DataFrame(columns=['entry_date', 'exit_date', 'entry_price',  # type: ignore[arg-type]
                                 'exit_price', 'size', 'symbol'])


# ==============================================================================
# THE KILL SWITCH
# ==============================================================================

class TestKillSwitchLimits(unittest.TestCase):

    def test_defaults_are_generated_not_typed(self):
        """
        Derived from firm_rules.ftmo(), so editing that profile moves these
        too rather than leaving a stale literal behind.
        """
        import kill_switch
        cfg = kill_switch.KillSwitchConfig()
        base = ftmo()
        self.assertAlmostEqual(cfg.ftmo_daily_limit_pct,
                               base.max_daily_loss_pct * 100)
        self.assertAlmostEqual(cfg.ftmo_total_limit_pct,
                               base.max_total_drawdown_pct * 100)

    def test_for_firm_matches_the_profile(self):
        import kill_switch
        cfg = kill_switch.KillSwitchConfig.for_firm(TIGHT)
        self.assertAlmostEqual(cfg.ftmo_daily_limit_pct, 3.0)
        self.assertAlmostEqual(cfg.ftmo_total_limit_pct, 6.0)

    def test_for_firm_enables_ftmo_mode(self):
        """Building a config for a firm and leaving the checks off is a
        configuration that looks protective and is not."""
        import kill_switch
        self.assertTrue(kill_switch.KillSwitchConfig.for_firm(TIGHT).ftmo_mode)

    def test_for_firm_allows_overrides(self):
        import kill_switch
        cfg = kill_switch.KillSwitchConfig.for_firm(TIGHT, ftmo_mode=False)
        self.assertFalse(cfg.ftmo_mode)

    def test_no_hardcoded_literals_remain(self):
        import kill_switch
        src = inspect.getsource(kill_switch.KillSwitchConfig)
        self.assertNotIn('ftmo_daily_limit_pct: float = 5.0', src)
        self.assertNotIn('ftmo_total_limit_pct: float = 10.0', src)


# ==============================================================================
# ONE PROFILE, EVERY COMPONENT
# ==============================================================================

class TestProfileReachesEveryComponent(unittest.TestCase):
    """
    The integration this is all for: one FirmRules, consistent numbers
    everywhere it lands.
    """

    def test_governor(self):
        from live_governor import GovernorConfig, LiveGovernor
        g = LiveGovernor(GovernorConfig(rules=TIGHT))
        self.assertAlmostEqual(g.config.rules.daily_loss_limit(100_000), 3_000)

    def test_kill_switch(self):
        import kill_switch
        cfg = kill_switch.KillSwitchConfig.for_firm(TIGHT)
        self.assertAlmostEqual(cfg.ftmo_daily_limit_pct, 3.0)

    def test_compliance_checker(self):
        from ftmo_compliance import FTMOComplianceChecker
        ck = FTMOComplianceChecker(rules=TIGHT)
        self.assertAlmostEqual(ck.rules.daily_loss_limit(100_000), 3_000)

    def test_challenge_simulator(self):
        from challenge_simulator import StageSpec
        st = StageSpec.from_rules(TIGHT, 'challenge')
        self.assertAlmostEqual(st.profit_target_pct, 0.08)
        self.assertEqual(st.min_trading_days, 7)

    def test_all_agree_on_the_daily_limit(self):
        """
        The one that would have caught the original bug: every component
        computing the same currency limit from the same profile.
        """
        from ftmo_compliance import FTMOComplianceChecker
        from live_governor import GovernorConfig, LiveGovernor
        import kill_switch

        expected = TIGHT.max_daily_loss_pct * 100_000     # 3,000

        checker = FTMOComplianceChecker(rules=TIGHT)
        governor = LiveGovernor(GovernorConfig(rules=TIGHT))
        ks = kill_switch.KillSwitchConfig.for_firm(TIGHT)

        self.assertAlmostEqual(checker.rules.daily_loss_limit(100_000), expected)
        self.assertAlmostEqual(
            governor.config.rules.daily_loss_limit(100_000), expected)
        self.assertAlmostEqual(
            ks.ftmo_daily_limit_pct / 100 * 100_000, expected)


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('\n' + '=' * 68)
    print(f"  ran {result.testsRun} | failures {len(result.failures)} | "
          f"errors {len(result.errors)} | skipped {len(result.skipped)}")
    print('=' * 68)
    if result.skipped:
        for case, reason in result.skipped:
            print(f"    SKIPPED {case}: {reason}")
    return 0 if not (result.failures or result.errors or result.skipped) else 1


if __name__ == '__main__':
    sys.exit(main())