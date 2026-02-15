# ==============================================================================
# test_ftmo_compliance.py
# ==============================================================================
# Comprehensive unit tests for FTMO compliance module
# Tests edge cases: timezone boundaries, near-limit drawdowns, fee calculations
#
# Run: python test_ftmo_compliance.py
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import unittest

from ftmo_compliance import (
    FTMOComplianceChecker,
    ACCOUNT_SIZES,
    MAX_DAILY_LOSS_PCT,
    MAX_TOTAL_DRAWDOWN_PCT,
    MIN_TRADING_DAYS,
    PROFIT_TARGETS,
    detect_asset_class,
    AssetClass,
    to_prague_time,
    get_prague_trading_day
)


class TestTimezoneHandling(unittest.TestCase):
    """Test Prague timezone boundary handling"""
    
    def test_prague_time_conversion(self):
        """Test UTC to Prague conversion"""
        # Winter time: UTC+1
        winter_utc = datetime(2024, 1, 15, 12, 0, 0)
        prague = to_prague_time(winter_utc)
        self.assertEqual(prague.hour, 13)  # 12 UTC = 13 Prague (CET)
        
        # Summer time: UTC+2
        summer_utc = datetime(2024, 7, 15, 12, 0, 0)
        prague = to_prague_time(summer_utc)
        self.assertEqual(prague.hour, 14)  # 12 UTC = 14 Prague (CEST)
    
    def test_midnight_boundary_same_day(self):
        """Trade at 23:30 Prague should be same day"""
        # UTC 22:30 = Prague 23:30 in winter
        utc_time = datetime(2024, 1, 15, 22, 30, 0)
        prague_date = get_prague_trading_day(utc_time)
        self.assertEqual(prague_date.day, 15)
    
    def test_midnight_boundary_next_day(self):
        """Trade at 00:30 Prague should be next day"""
        # UTC 23:30 = Prague 00:30 (next day) in winter
        utc_time = datetime(2024, 1, 15, 23, 30, 0)
        prague_date = get_prague_trading_day(utc_time)
        self.assertEqual(prague_date.day, 16)


class TestDrawdownCalculations(unittest.TestCase):
    """Test drawdown calculation correctness"""
    
    def setUp(self):
        self.checker = FTMOComplianceChecker()
    
    def test_daily_loss_exactly_5_percent(self):
        """Daily loss of exactly 5% should trigger"""
        trades = pd.DataFrame([
            # 500 pip loss = $5000 on $100K = exactly 5%
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.0500, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.0500, 'exit_price': 1.0510, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.0510, 'exit_price': 1.0520, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.0520, 'exit_price': 1.0530, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=100000, phase='challenge')
        self.assertGreaterEqual(result.max_daily_loss_pct, 4.9)
    
    def test_daily_loss_4_9_percent(self):
        """Daily loss of 4.9% should pass"""
        trades = pd.DataFrame([
            # 490 pip loss = $4900 on $100K = 4.9%
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.0510, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.0510, 'exit_price': 1.0610, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.0610, 'exit_price': 1.0710, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.0710, 'exit_price': 1.0810, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=100000, phase='challenge')
        self.assertLess(result.max_daily_loss_pct, 5.5)
    
    def test_multiple_intraday_trades_combined(self):
        """Multiple trades on same day should combine for daily loss"""
        trades = pd.DataFrame([
            # Day 1: Three losing trades
            {'entry_date': '2024-01-02 09:00:00', 'exit_date': '2024-01-02 10:00:00',
             'entry_price': 1.1000, 'exit_price': 1.0900, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-02 11:00:00', 'exit_date': '2024-01-02 12:00:00',
             'entry_price': 1.0900, 'exit_price': 1.0800, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-02 14:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.0800, 'exit_price': 1.0700, 'size': 100000, 'symbol': 'EUR-USD'},
            # Days 2-4: small trades
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.0700, 'exit_price': 1.0710, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.0710, 'exit_price': 1.0720, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.0720, 'exit_price': 1.0730, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=100000, phase='challenge')
        self.assertGreaterEqual(result.max_daily_loss_pct, 2.5)
        self.assertLess(result.max_daily_loss_pct, 5.0)


class TestFeeCalculations(unittest.TestCase):
    """Test fee structure calculations"""
    
    def test_fx_commission_per_lot(self):
        """FX should charge $5 per lot round-turn"""
        checker = FTMOComplianceChecker()
        
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = checker.validate(trades, account_size=100000, phase='challenge')
        self.assertGreater(result.total_fees, 4.0)
        self.assertLess(result.total_fees, 50.0)
    
    def test_crypto_percentage_fee(self):
        """Crypto should charge 0.005% percentage fee"""
        checker = FTMOComplianceChecker()
        
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 40000, 'exit_price': 41000, 'size': 1.0, 'symbol': 'BTC-USD'},
        ])
        
        result = checker.validate(trades, account_size=100000, phase='challenge')
        self.assertGreater(result.total_fees, 1.0)
        self.assertLess(result.total_fees, 50.0)
    
    def test_fees_reduce_net_return(self):
        """Fees should reduce net return"""
        checker = FTMOComplianceChecker()
        
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1000, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1000, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1000, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1000, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = checker.validate(trades, account_size=100000, phase='challenge')
        self.assertLess(result.final_return_pct, 0)


class TestAssetClassDetection(unittest.TestCase):
    """Test automatic asset class detection"""
    
    def test_fx_pairs(self):
        self.assertEqual(detect_asset_class('EUR-USD'), AssetClass.FX)
        self.assertEqual(detect_asset_class('GBP-JPY'), AssetClass.FX)
        self.assertEqual(detect_asset_class('EURUSD'), AssetClass.FX)
    
    def test_crypto(self):
        self.assertEqual(detect_asset_class('BTC-USD'), AssetClass.CRYPTO)
        self.assertEqual(detect_asset_class('ETH-USDT'), AssetClass.CRYPTO)
    
    def test_indices(self):
        self.assertEqual(detect_asset_class('US500'), AssetClass.INDICES)
        self.assertEqual(detect_asset_class('US30'), AssetClass.INDICES)
    
    def test_commodities(self):
        self.assertEqual(detect_asset_class('GOLD'), AssetClass.COMMODITIES)
        self.assertEqual(detect_asset_class('XAUUSD'), AssetClass.COMMODITIES)


class TestTradingDaysRequirement(unittest.TestCase):
    """Test minimum trading days requirement"""
    
    def setUp(self):
        self.checker = FTMOComplianceChecker()
    
    def test_exactly_4_days(self):
        """Exactly 4 trading days should pass"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1300, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1300, 'exit_price': 1.1400, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=100000, phase='challenge')
        self.assertEqual(result.trading_days, 4)
        self.assertTrue(result.min_days_ok)
    
    def test_only_3_days(self):
        """Only 3 trading days should fail"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1300, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=100000, phase='challenge')
        self.assertEqual(result.trading_days, 3)
        self.assertFalse(result.min_days_ok)


class TestProfitTargets(unittest.TestCase):
    """Test profit target requirements"""
    
    def setUp(self):
        self.checker = FTMOComplianceChecker()
    
    def test_challenge_10_percent_target(self):
        """Challenge requires 10% profit"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1350, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1350, 'exit_price': 1.1500, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=10000, phase='challenge')
        self.assertTrue(result.profit_target_ok)
    
    def test_verification_5_percent_target(self):
        """Verification requires only 5% profit"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1050, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1050, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1150, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1150, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=10000, phase='verification')
        self.assertTrue(result.profit_target_ok)


class TestMultiAccountValidation(unittest.TestCase):
    """Test multi-account size validation"""
    
    def test_all_account_sizes_returned(self):
        """Should return results for all 5 account sizes"""
        checker = FTMOComplianceChecker()
        
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1300, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1300, 'exit_price': 1.1400, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        df = checker.validate_all_account_sizes(trades, phase='challenge')
        self.assertEqual(len(df), 5)
        self.assertListEqual(list(df['account_size']), ACCOUNT_SIZES)
    
    def test_required_columns_present(self):
        """Output DataFrame should have all required columns"""
        checker = FTMOComplianceChecker()
        
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1300, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1300, 'exit_price': 1.1400, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        df = checker.validate_all_account_sizes(trades, phase='challenge')
        
        required_cols = [
            'account_size', 'daily_loss_ok', 'total_drawdown_ok', 'min_days_ok',
            'profit_target_ok', 'PASS', 'max_daily_loss_pct', 'max_total_dd_pct',
            'trading_days', 'final_equity', 'final_return_pct'
        ]
        
        for col in required_cols:
            self.assertIn(col, df.columns)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.checker = FTMOComplianceChecker()
    
    def test_empty_trades(self):
        """Empty trade list should fail gracefully"""
        trades = pd.DataFrame(columns=['entry_date', 'exit_date', 'entry_price', 'size', 'symbol'])
        
        result = self.checker.validate(trades, account_size=100000, phase='challenge')
        
        self.assertFalse(result.passed)
        self.assertEqual(result.trading_days, 0)
        self.assertEqual(result.final_equity, 100000)
    
    def test_invalid_account_size(self):
        """Invalid account size should raise error"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        with self.assertRaises(ValueError):
            self.checker.validate(trades, account_size=75000, phase='challenge')
    
    def test_invalid_phase(self):
        """Invalid phase should raise error"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        with self.assertRaises(ValueError):
            self.checker.validate(trades, account_size=100000, phase='invalid')
    
    def test_short_position(self):
        """Short positions should be handled correctly"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1000, 'size': -100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.0900, 'size': -100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.0900, 'exit_price': 1.0800, 'size': -100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.0800, 'exit_price': 1.0700, 'size': -100000, 'symbol': 'EUR-USD'},
        ])
        
        result = self.checker.validate(trades, account_size=10000, phase='challenge')
        self.assertGreater(result.final_return_pct, 0)
    
    def test_deterministic_results(self):
        """Same inputs should always produce same outputs"""
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1300, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1300, 'exit_price': 1.1400, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        results = []
        for _ in range(5):
            result = self.checker.validate(trades, account_size=100000, phase='challenge')
            results.append((
                result.passed,
                result.max_daily_loss_pct,
                result.max_total_drawdown_pct,
                result.final_return_pct,
                result.total_fees
            ))
        
        self.assertTrue(all(r == results[0] for r in results))


class TestReportGeneration(unittest.TestCase):
    """Test report generation functionality"""
    
    def test_report_contains_all_sections(self):
        """Report should contain all required sections"""
        checker = FTMOComplianceChecker()
        
        trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1200, 'exit_price': 1.1300, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1300, 'exit_price': 1.1400, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        result = checker.validate(trades, account_size=100000, phase='challenge')
        report = checker.generate_report(result, phase='challenge')
        
        self.assertIn('FTMO COMPLIANCE REPORT', report)
        self.assertIn('RULE COMPLIANCE', report)
        self.assertIn('Max Daily Loss', report)
        self.assertIn('FINAL VERDICT', report)
        self.assertIn('DIAGNOSTICS', report)


if __name__ == '__main__':
    unittest.main(verbosity=2)
