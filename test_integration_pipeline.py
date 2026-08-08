# ==============================================================================
# test_integration_pipeline.py
# ==============================================================================
# CONTRACTS BETWEEN MODULES, not module internals.
#
# 345 unit tests each check one module against a fixture built to suit it.
# None check that module A's real output satisfies module B's real input --
# which is how bootstrap_summary and challenge_simulator ended up disagreeing
# about P(pass) by a factor of two while both suites stayed green.
#
# Every test here runs real output from one component into the next.
# ==============================================================================

import sys
import unittest
from datetime import timedelta
from typing import Optional, TypeVar

import numpy as np
import pandas as pd

from broker_adapter import BaseBroker
from canonical_result import CanonicalResult
import challenge_simulator
import consistency_rule
import dashboard_portfolio_panel as PANEL
import portfolio_merge
from challenge_simulator import simulate_challenge
from firm_rules import Capability, FirmRules, ftmo, generic_trailing
from ftmo_compliance import FTMOComplianceChecker
from governed_broker import GovernedBroker, account_state_from_broker
from live_governor import Decision, GovernorConfig, LiveGovernor
from portfolio_merge import joint_block_bootstrap, merge_strategies

ACCOUNT = 100_000.0

_T = TypeVar('_T')


def not_none(v: Optional[_T], msg: str = 'expected a value') -> _T:
    assert v is not None, msg
    return v


def strategy(sid, pnls, size=10_000.0, start='2024-01-01', symbol='EUR-USD'):
    """
    A CanonicalResult whose prices are CONSISTENT with its P&L.

    The consistency matters: the compliance checker recomputes P&L from
    entry/exit prices and size, so a fixture whose stated pnl disagrees with
    its prices tests the warning path rather than the contract.
    """
    base = pd.Timestamp(start)
    trades = []
    for i, p in enumerate(pnls):
        ex = base + timedelta(days=i, hours=15)
        trades.append({
            'entry_date': (ex - timedelta(hours=2)).isoformat(),
            'exit_date': ex.isoformat(),
            'entry_price': 1.1000,
            'exit_price': 1.1000 + p / size,
            'size': size,
            'symbol': symbol,
            'pnl': float(p),
        })
    cr = CanonicalResult(
        strategy_id=sid, strategy_name=sid, symbol=symbol, timeframe='M15',
        starting_value=ACCOUNT, total_trades=len(trades), trade_list=trades)
    cr._compute_arrays()
    return cr


A_PNL = [1200, -800, 500, 800, -200, 900, -400, 600, 300, -150]
B_PNL = [-400, 900, 1100, -300, 700, -1200, 400, 500, -250, 800]


# ==============================================================================
# MERGE -> COMPLIANCE CHECKER
# ==============================================================================

class TestMergedResultThroughChecker(unittest.TestCase):
    """
    The central Phase 3 claim: a merged portfolio is an ordinary
    CanonicalResult and flows through the identical validation pipeline.
    Nothing tested that until now.
    """

    def setUp(self):
        self.res = merge_strategies(
            [strategy('A', A_PNL), strategy('B', B_PNL)],
            rules=ftmo(), account_size=ACCOUNT)
        self.ledger = pd.DataFrame(self.res.canonical.trade_list)
        self.checker = FTMOComplianceChecker(rules=ftmo())

    def test_checker_accepts_the_merged_ledger(self):
        result = self.checker.validate(self.ledger, account_size=100_000)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.passed, bool)

    def test_ledger_carries_every_column_the_checker_needs(self):
        for col in ('entry_date', 'exit_date', 'entry_price', 'exit_price',
                    'size', 'symbol'):
            self.assertIn(col, self.ledger.columns)

    def test_merge_total_is_gross_and_checker_total_is_net(self):
        """
        These two numbers DIFFER BY DESIGN and the difference is the fees.

        The merge sums raw trade P&L; the checker applies commission and
        spread. Comparing them without accounting for that looks like a bug
        and is not one -- pinning it here so the next person to notice finds
        an explanation rather than a mystery.
        """
        merge_total = sum(t['pnl'] for t in self.res.canonical.trade_list)
        r = self.checker.validate(self.ledger, account_size=100_000)
        self.assertGreater(r.total_fees, 0, 'checker should model costs')
        self.assertAlmostEqual(merge_total, r.total_pnl + r.total_fees,
                               places=2)
        self.assertLess(r.total_pnl, merge_total)

    def test_checker_recomputes_pnl_and_wins(self):
        """
        The checker derives P&L from prices and IGNORES the pnl column.

        For a ledger whose prices imply its P&L this is invisible. For one
        carrying cost-inclusive or multi-leg P&L, the checker silently
        substitutes its own number. It prints a warning; nothing captures it.
        """
        bad = self.ledger.copy()
        bad['pnl'] = bad['pnl'] * 10          # prices unchanged
        r_bad = self.checker.validate(bad, account_size=100_000)
        r_good = self.checker.validate(self.ledger, account_size=100_000)
        self.assertAlmostEqual(r_bad.total_pnl, r_good.total_pnl, places=6)

    def test_trade_count_survives_the_round_trip(self):
        r = self.checker.validate(self.ledger, account_size=100_000)
        self.assertEqual(len(self.ledger), len(A_PNL) + len(B_PNL))
        self.assertGreater(r.trading_days, 0)


# ==============================================================================
# MERGE -> BOOTSTRAP -> CHALLENGE SIMULATOR
# ==============================================================================

class TestMergeIntoSimulator(unittest.TestCase):

    def setUp(self):
        self.res = merge_strategies(
            [strategy('A', A_PNL * 8), strategy('B', B_PNL * 8)],
            rules=ftmo(), account_size=ACCOUNT)

    def test_daily_matrix_feeds_the_bootstrap(self):
        sims = joint_block_bootstrap(self.res.daily_pnl, n_simulations=200,
                                     window_days=30)
        self.assertEqual(sims.shape, (200, 30))

    def test_bootstrap_feeds_the_challenge_simulator(self):
        sims = joint_block_bootstrap(self.res.daily_pnl, n_simulations=300,
                                     window_days=30, random_seed=3)
        out = simulate_challenge(sims, ACCOUNT, ftmo())
        self.assertEqual(out.n_simulations, 300)
        self.assertGreaterEqual(out.p_funded, 0.0)
        self.assertLessEqual(out.p_funded, 1.0)

    def test_panel_wrapper_runs_the_same_chain(self):
        out = PANEL.try_challenge(self.res.daily_pnl, ftmo(), ACCOUNT,
                                  n_simulations=200, window_days=30)
        self.assertTrue(out['ok'], out.get('reason'))
        self.assertEqual(len(out['stages']), 2)

    def test_bootstrap_and_simulator_agree_on_one_stage(self):
        """
        The regression that started this: two components computing P(pass)
        from the same paths must not diverge.
        """
        sims = joint_block_bootstrap(self.res.daily_pnl, n_simulations=400,
                                     window_days=30, random_seed=7)
        summary = portfolio_merge.bootstrap_summary(sims, ACCOUNT, ftmo())
        stage = challenge_simulator.StageSpec.from_rules(ftmo(), 'challenge')
        walked = sum(
            1 for row in sims
            if challenge_simulator.walk_stage(
                row, ACCOUNT, stage, ftmo())['outcome']
            == challenge_simulator.PASSED) / len(sims)
        self.assertAlmostEqual(summary['modelled_pass_rate'], walked,
                               places=12)


# ==============================================================================
# MERGE -> COMPARISON TABLE
# ==============================================================================

class TestMergeIntoComparison(unittest.TestCase):

    def test_portfolio_renders_beside_its_members(self):
        a, b = strategy('A', A_PNL), strategy('B', B_PNL)
        res = merge_strategies([a, b], rules=ftmo(), account_size=ACCOUNT)
        table = PANEL.build_comparison([a, b], portfolio=res.canonical)
        self.assertEqual(len(table.columns), 3)
        self.assertTrue(table.columns[-1].is_portfolio)
        self.assertTrue(table.deltas)

    def test_merged_result_has_real_provenance(self):
        """A portfolio must earn trade_list provenance like anything else."""
        res = merge_strategies(
            [strategy('A', A_PNL), strategy('B', B_PNL)],
            rules=ftmo(), account_size=ACCOUNT)
        self.assertEqual(res.canonical.returns_source, 'trade_list')
        res.canonical.require_returns('integration test')


# ==============================================================================
# BROKER -> GOVERNOR -> GOVERNED BROKER
# ==============================================================================

class MiniBroker(BaseBroker):
    """
    Smallest real BaseBroker satisfying what the state bridge reads.

    Subclasses BaseBroker rather than duck-typing it: the point of these
    tests is that the actual contract holds, and a stand-in that only
    resembles the interface would not prove that.
    """

    def __init__(self, equity, unrealized=0.0):
        super().__init__(name='mini')
        self.is_connected = True
        self.equity = equity
        self.unrealized = unrealized
        self.submitted = []

    def connect(self):
        return True

    def disconnect(self):
        self.is_connected = False

    def get_balance(self):
        from broker_adapter import BrokerBalance
        return BrokerBalance(total_equity=self.equity, free_margin=self.equity,
                             used_margin=0.0, unrealized_pnl=self.unrealized,
                             currency='USD', timestamp='')

    def get_positions(self):
        return []

    def get_position(self, symbol):
        return None

    def get_tick(self, symbol):
        return None

    def get_order(self, order_id):
        return None

    def cancel_order(self, order_id):
        return True

    def submit_order(self, side, symbol, size, order_type='market',
                     price=None, stop_price=None):
        from broker_adapter import BrokerOrder, OrderSide, OrderStatus, OrderType
        self.submitted.append((side, symbol, size))
        return BrokerOrder(
            order_id='1', symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET, size=size,
            status=OrderStatus.FILLED, timestamp='')


class TestBrokerThroughGovernor(unittest.TestCase):

    def test_state_bridge_produces_a_usable_verdict(self):
        broker = MiniBroker(equity=99_000)
        state = account_state_from_broker(broker, ACCOUNT)
        g = LiveGovernor()
        g.seed_anchor(g.trading_date(state.timestamp), ACCOUNT)
        self.assertIs(g.observe(state).decision, Decision.ALLOW)

    def test_full_chain_blocks_an_order(self):
        broker = MiniBroker(equity=95_800)
        g = LiveGovernor()
        g.seed_anchor(g.trading_date(
            account_state_from_broker(broker, ACCOUNT).timestamp), ACCOUNT)
        gb = GovernedBroker(broker, g, initial_balance=ACCOUNT)
        from broker_adapter import OrderStatus
        o = gb.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(broker.submitted, [])

    def test_full_chain_allows_a_healthy_order(self):
        broker = MiniBroker(equity=ACCOUNT)
        g = LiveGovernor()
        g.seed_anchor(g.trading_date(
            account_state_from_broker(broker, ACCOUNT).timestamp), ACCOUNT)
        gb = GovernedBroker(broker, g, initial_balance=ACCOUNT)
        gb.submit_order('buy', 'EURUSD', 100_000)
        self.assertEqual(len(broker.submitted), 1)


# ==============================================================================
# ONE PROFILE, END TO END
# ==============================================================================

class TestUncheckedRulesPropagate(unittest.TestCase):
    """
    A partial rule set must stay visible the whole way. Any component that
    drops the list turns a qualified answer into an unqualified one.
    """

    def setUp(self):
        self.rules = generic_trailing('PartialFirm')
        self.res = merge_strategies(
            [strategy('A', A_PNL * 6), strategy('B', B_PNL * 6)],
            rules=self.rules, account_size=ACCOUNT)

    def test_merge_diagnostics(self):
        caps = [u.capability.value for u in self.res.diagnostics.unsupported_rules]
        self.assertIn('trailing_drawdown_eod', caps)

    def test_bootstrap_summary(self):
        sims = joint_block_bootstrap(self.res.daily_pnl, n_simulations=100,
                                     window_days=20)
        s = portfolio_merge.bootstrap_summary(sims, ACCOUNT, self.rules)
        self.assertIn('trailing_drawdown_eod', s['unsupported_rules'])
        self.assertFalse(s['is_complete'])

    def test_challenge_result(self):
        sims = joint_block_bootstrap(self.res.daily_pnl, n_simulations=100,
                                     window_days=20)
        out = simulate_challenge(sims, ACCOUNT, self.rules)
        self.assertIn('trailing_drawdown_eod', out.unchecked_rules)

    def test_governor_verdict(self):
        g = LiveGovernor(GovernorConfig(rules=self.rules))
        broker = MiniBroker(equity=99_000)
        state = account_state_from_broker(broker, ACCOUNT)
        g.seed_anchor(g.trading_date(state.timestamp), ACCOUNT)
        self.assertIn('trailing_drawdown_eod',
                      g.observe(state).unchecked_rules)

    def test_governed_broker_summary(self):
        broker = MiniBroker(equity=99_000)
        g = LiveGovernor(GovernorConfig(rules=self.rules))
        g.seed_anchor(g.trading_date(
            account_state_from_broker(broker, ACCOUNT).timestamp), ACCOUNT)
        gb = GovernedBroker(broker, g, initial_balance=ACCOUNT)
        gb.submit_order('buy', 'EURUSD', 100_000)
        self.assertIn('trailing_drawdown_eod',
                      gb.summary()['unchecked_rules'])


class TestConsistencyEndToEnd(unittest.TestCase):

    def test_cap_reduces_p_funded_through_the_whole_chain(self):
        res = merge_strategies(
            [strategy('A', A_PNL * 8), strategy('B', B_PNL * 8)],
            rules=ftmo(), account_size=ACCOUNT)
        sims = joint_block_bootstrap(res.daily_pnl, n_simulations=800,
                                     window_days=40, random_seed=11)
        loose = simulate_challenge(sims, ACCOUNT, ftmo())
        strict = simulate_challenge(
            sims, ACCOUNT,
            FirmRules(firm_name='Capped', consistency_max_day_pct=0.25))
        self.assertLessEqual(strict.p_funded, loose.p_funded)


# ==============================================================================
# SMOKE
# ==============================================================================

class TestFullChainSmoke(unittest.TestCase):

    def test_strategies_to_p_funded_without_exceptions(self):
        rules = ftmo()
        res = merge_strategies(
            [strategy('A', A_PNL * 10), strategy('B', B_PNL * 10),
             strategy('C', [x * 0.5 for x in A_PNL] * 10)],
            rules=rules, account_size=ACCOUNT)

        checker = FTMOComplianceChecker(rules=rules)
        compliance = checker.validate(
            pd.DataFrame(res.canonical.trade_list), account_size=100_000)

        sims = joint_block_bootstrap(res.daily_pnl, n_simulations=500,
                                     window_days=40, random_seed=5)
        challenge = simulate_challenge(sims, ACCOUNT, rules)
        cons = consistency_rule.check_consistency_frame(res.daily_pnl, 0.30)

        self.assertEqual(len(res.diagnostics.strategy_ids), 3)
        self.assertIsInstance(compliance.passed, bool)
        self.assertGreaterEqual(challenge.p_funded, 0.0)
        self.assertIsNotNone(cons)

    def test_single_strategy_is_refused_by_the_merge(self):
        """A portfolio of one is a strategy; say so rather than degrade."""
        with self.assertRaises(portfolio_merge.PortfolioMergeError):
            merge_strategies([strategy('A', A_PNL)], rules=ftmo())


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
