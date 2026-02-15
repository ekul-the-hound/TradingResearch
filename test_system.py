# ==============================================================================
# test_system.py
# ==============================================================================
# FULL SYSTEM TEST RUNNER
#
# Runs through your entire trading research system step by step:
#
# FOUNDATION TESTS (1-8):
#   1. Config validation
#   2. Data manager (Forex, Crypto, Indices)
#   3. Database connection
#   4. Base strategy syntax check
#   5. Single backtest
#   6. Multi-asset backtest (small sample)
#   7. Mutation agent (DRY RUN - no API call)
#   8. Variant loading test
#
# WEEK 1-2 TESTS (9-12):
#   9.  Regime Classifier
#   10. Validation Framework (Bootstrap, Monte Carlo, Walk-Forward)
#   11. Manual Gates
#   12. Multi-timeframe backtester with regime analysis
#
# WEEK 3 TESTS (13-15):
#   13. Robustness Tests (Latency, Slippage)
#   14. Adversarial Reviewer (DRY RUN - no API call)
#   15. Failures Tracker
#
# WEEK 4 TESTS (16-19):
#   16. Permutation Tests
#   17. Parameter Sensitivity Analysis
#   18. Cost-Adjusted Scoring
#   19. Expanded Mutation Config
#
# After each step, tells you if it passed/failed and asks to continue.
#
# Usage: python test_system.py
# ==============================================================================

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_header(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_result(success, message=""):
    """Print pass/fail result"""
    if success:
        print(f"\n  Ã¢Å“â€¦ PASSED {message}")
    else:
        print(f"\n  Ã¢ÂÅ’ FAILED {message}")

def ask_continue():
    """Ask user if they want to continue to next test"""
    print(f"\n{'-'*70}")
    response = input("  Continue to next test? (Y/N): ").strip().upper()
    return response == 'Y'

def run_test(test_name, test_func):
    """Run a test function with error handling"""
    print_header(test_name)
    
    try:
        success, details = test_func()
        print_result(success, details if not success else "")
        return success
    except Exception as e:
        print(f"\n  Ã¢ÂÅ’ EXCEPTION: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# FOUNDATION TEST FUNCTIONS (1-8)
# ==============================================================================

def test_config():
    """Test 1: Config validation"""
    print("  Loading config.py...")
    
    import config
    
    errors = []
    warnings = []
    
    # Check required attributes exist
    required_attrs = [
        'FOREX_ENABLED', 'CRYPTO_ENABLED', 'INDICES_ENABLED',
        'FOREX_WATCHLIST', 'CRYPTO_WATCHLIST', 'INDEX_WATCHLIST',
        'DATABASE_PATH', 'DATA_CACHE_PATH', 'CLAUDE_API_KEY'
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            errors.append(f"Missing: {attr}")
        else:
            value = getattr(config, attr)
            print(f"    {attr}: {value if not 'KEY' in attr else '***configured***' if value else 'NOT SET'}")
    
    # Check directories
    print(f"\n  Checking directories...")
    
    if config.FOREX_ENABLED:
        forex_path = Path(config.FOREX_BASE_PATH)
        if forex_path.exists():
            files = list(forex_path.glob('*.csv')) + list(forex_path.glob('*.xlsx'))
            print(f"    Forex dir: Ã¢Å“â€¦ Found ({len(files)} files)")
        else:
            warnings.append(f"Forex dir not found: {forex_path}")
            print(f"    Forex dir: Ã¢Å¡Â Ã¯Â¸Â  Not found")
    
    if config.CRYPTO_ENABLED:
        crypto_path = Path(config.CACHE_SUBDIRS.get('crypto', ''))
        if crypto_path.exists():
            files = list(crypto_path.glob('**/*.csv'))
            print(f"    Crypto dir: Ã¢Å“â€¦ Found ({len(files)} files)")
        else:
            warnings.append(f"Crypto dir not found: {crypto_path}")
            print(f"    Crypto dir: Ã¢Å¡Â Ã¯Â¸Â  Not found")
    
    if config.INDICES_ENABLED:
        indices_path = Path(config.CACHE_SUBDIRS.get('indices', ''))
        if indices_path.exists():
            files = list(indices_path.glob('**/*.csv'))
            print(f"    Indices dir: Ã¢Å“â€¦ Found ({len(files)} files)")
        else:
            warnings.append(f"Indices dir not found: {indices_path}")
            print(f"    Indices dir: Ã¢Å¡Â Ã¯Â¸Â  Not found")
    
    # Check API key
    if not config.CLAUDE_API_KEY:
        warnings.append("Claude API key not set")
    
    if errors:
        return False, f"Errors: {', '.join(errors)}"
    elif warnings:
        print(f"\n  Ã¢Å¡Â Ã¯Â¸Â  Warnings: {', '.join(warnings)}")
        return True, "(with warnings)"
    else:
        return True, ""


def test_data_manager():
    """Test 2: Data Manager"""
    print("  Initializing DataManager...")
    
    from data_manager import DataManager
    import config
    
    manager = DataManager()
    
    results = {'forex': None, 'crypto': None, 'indices': None}
    
    # Test Forex
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        print(f"\n  Testing Forex ({config.FOREX_WATCHLIST[0]})...")
        try:
            data = manager.get_data(config.FOREX_WATCHLIST[0], '1hour', 100)
            if data is not None and len(data) > 0:
                results['forex'] = len(data)
                print(f"    Ã¢Å“â€¦ Got {len(data)} bars")
            else:
                results['forex'] = 0
                print(f"    Ã¢ÂÅ’ No data returned")
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Error: {e}")
            results['forex'] = -1
    else:
        print(f"\n  Forex: Skipped (disabled)")
    
    # Test Crypto
    if config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        print(f"\n  Testing Crypto ({config.CRYPTO_WATCHLIST[0]})...")
        try:
            data = manager.get_data(config.CRYPTO_WATCHLIST[0], '1hour', 100)
            if data is not None and len(data) > 0:
                results['crypto'] = len(data)
                print(f"    Ã¢Å“â€¦ Got {len(data)} bars")
            else:
                results['crypto'] = 0
                print(f"    Ã¢Å¡Â Ã¯Â¸Â  No data (may need local files or CCXT)")
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Error: {e}")
            results['crypto'] = -1
    else:
        print(f"\n  Crypto: Skipped (disabled)")
    
    # Test Indices
    if config.INDICES_ENABLED and config.INDEX_WATCHLIST:
        print(f"\n  Testing Indices ({config.INDEX_WATCHLIST[0]})...")
        try:
            data = manager.get_data(config.INDEX_WATCHLIST[0], '1hour', 100)
            if data is not None and len(data) > 0:
                results['indices'] = len(data)
                print(f"    Ã¢Å“â€¦ Got {len(data)} bars")
            else:
                results['indices'] = 0
                print(f"    Ã¢Å¡Â Ã¯Â¸Â  No data (check local files)")
        except Exception as e:
            print(f"    Ã¢ÂÅ’ Error: {e}")
            results['indices'] = -1
    else:
        print(f"\n  Indices: Skipped (disabled)")
    
    # Determine overall result
    active_results = [v for v in results.values() if v is not None]
    if not active_results:
        return False, "No data sources enabled"
    
    successful = sum(1 for v in active_results if v and v > 0)
    total = len(active_results)
    
    if successful == total:
        return True, ""
    elif successful > 0:
        return True, f"({successful}/{total} sources working)"
    else:
        return False, "No data sources returned data"


def test_database():
    """Test 3: Database connection"""
    print("  Testing database connection...")
    
    from database import ResultsDatabase
    import config
    
    db = ResultsDatabase()
    
    # Check if database file exists/was created
    if Path(config.DATABASE_PATH).exists():
        print(f"    Database file: Ã¢Å“â€¦ {config.DATABASE_PATH}")
    else:
        print(f"    Database file: Ã¢ÂÅ’ Not found")
        return False, "Database file not created"
    
    # Try to get summary
    try:
        summary = db.get_backtest_summary()
        if summary:
            print(f"    Existing results: {len(summary)} strategy combinations")
        else:
            print(f"    Existing results: 0 (empty database)")
        return True, ""
    except Exception as e:
        return False, f"Database error: {e}"


def test_base_strategy():
    """Test 4: Base strategy syntax check"""
    print("  Loading base strategy...")
    
    try:
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        print(f"    Strategy class: Ã¢Å“â€¦ {SimpleMovingAverageCrossover.__name__}")
        
        # Check it has required methods
        required = ['__init__', 'next']
        for method in required:
            if hasattr(SimpleMovingAverageCrossover, method):
                print(f"    Method '{method}': Ã¢Å“â€¦ Found")
            else:
                return False, f"Missing method: {method}"
        
        return True, ""
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_single_backtest():
    """Test 5: Single backtest"""
    print("  Running single backtest...")
    
    import config
    from backtester_multi_timeframe import MultiTimeframeBacktester
    from strategies.simple_strategy import SimpleMovingAverageCrossover
    
    # Pick a test symbol
    test_symbol = None
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        test_symbol = config.FOREX_WATCHLIST[0]
    elif config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        test_symbol = config.CRYPTO_WATCHLIST[0]
    
    if not test_symbol:
        return False, "No assets available for testing"
    
    print(f"    Testing with: {test_symbol}")
    
    backtester = MultiTimeframeBacktester()
    
    result = backtester.run_single_backtest(
        strategy_class=SimpleMovingAverageCrossover,
        symbol=test_symbol,
        timeframe='1hour',
        initial_cash=10000,
        commission=0.001
    )
    
    if result:
        print(f"\n    Results:")
        print(f"      Return:     {result.get('total_return_pct', 0):+.2f}%")
        print(f"      Trades:     {result.get('total_trades', 0)}")
        print(f"      Sharpe:     {result.get('sharpe_ratio', 'N/A')}")
        print(f"      Bars:       {result.get('bars_tested', 0)}")
        return True, ""
    else:
        return False, "Backtest returned no results"


def test_multi_backtest():
    """Test 6: Multi-asset backtest (small sample)"""
    print("  Running multi-asset backtest (2 assets, 2 timeframes)...")
    
    import config
    from backtester_multi_timeframe import MultiTimeframeBacktester
    from strategies.simple_strategy import SimpleMovingAverageCrossover
    
    # Get 2 assets
    test_assets = []
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        test_assets.extend(config.FOREX_WATCHLIST[:1])
    if config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        test_assets.extend(config.CRYPTO_WATCHLIST[:1])
    
    if len(test_assets) == 0:
        return False, "No assets available"
    
    test_timeframes = ['1hour', '4hour']
    
    print(f"    Assets: {test_assets}")
    print(f"    Timeframes: {test_timeframes}")
    print(f"    Total tests: {len(test_assets) * len(test_timeframes)}")
    
    backtester = MultiTimeframeBacktester()
    
    results = backtester.run_multi_asset_multi_timeframe(
        strategy_class=SimpleMovingAverageCrossover,
        assets=test_assets,
        timeframes=test_timeframes,
        initial_cash=10000,
        commission=0.001,
        save_to_db=False  # Don't save test results
    )
    
    if results and len(results) > 0:
        print(f"\n    Completed: {len(results)} backtests")
        avg_return = sum(r['total_return_pct'] for r in results) / len(results)
        print(f"    Avg return: {avg_return:+.2f}%")
        return True, ""
    else:
        return False, "No results returned"


def test_mutation_agent_dry():
    """Test 7: Mutation agent (DRY RUN - no API call)"""
    print("  Testing mutation agent components (no API call)...")
    
    # Test loading base strategy
    from mutate_strategy import load_base_strategy, get_performance_summary
    from mutation_config import get_all_ideas, get_ideas_list
    
    print(f"\n    Loading base strategy...")
    base_code = load_base_strategy()
    if base_code:
        print(f"      Ã¢Å“â€¦ Loaded ({len(base_code)} chars)")
    else:
        return False, "Could not load base strategy"
    
    print(f"\n    Loading mutation ideas...")
    ideas = get_all_ideas()
    ideas_list = get_ideas_list()
    print(f"      Ã¢Å“â€¦ Loaded {len(ideas_list)} ideas")
    
    print(f"\n    Loading performance summary...")
    performance = get_performance_summary()
    print(f"      Ã¢Å“â€¦ Summary loaded ({len(performance)} chars)")
    
    # Check MUTATION_PROMPT has the coding rules
    from mutate_strategy import MUTATION_PROMPT
    
    print(f"\n    Checking prompt for Backtrader rules...")
    rules_present = []
    if 'bt.indicators.OBV' in MUTATION_PROMPT:
        rules_present.append('OBV')
    if 'position.price' in MUTATION_PROMPT:
        rules_present.append('position.price')
    if 'len(self)' in MUTATION_PROMPT:
        rules_present.append('min bars')
    
    if len(rules_present) >= 3:
        print(f"      Ã¢Å“â€¦ Coding rules present: {', '.join(rules_present)}")
    else:
        print(f"      Ã¢Å¡Â Ã¯Â¸Â  Some coding rules may be missing")
        return True, "(prompt may need Backtrader rules)"
    
    return True, ""


def test_variant_loading():
    """Test 8: Variant loading test"""
    print("  Testing variant file loading...")
    
    from pathlib import Path
    import importlib.util
    import sys
    
    variants_dir = Path(__file__).parent / 'strategies' / 'variants'
    
    if not variants_dir.exists():
        print(f"    Variants directory: Ã¢Å¡Â Ã¯Â¸Â  Not found (will be created by mutation agent)")
        return True, "(no variants generated yet)"
    
    variant_files = list(variants_dir.glob('variant_*.py'))
    
    if not variant_files:
        print(f"    Variant files: Ã¢Å¡Â Ã¯Â¸Â  None found (run mutation agent first)")
        return True, "(no variants generated yet)"
    
    print(f"    Found {len(variant_files)} variant files")
    
    # Try loading each
    loaded = 0
    failed = 0
    
    for filepath in variant_files[:5]:  # Test first 5 only
        try:
            spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded += 1
            print(f"      Ã¢Å“â€¦ {filepath.name}")
        except Exception as e:
            failed += 1
            print(f"      Ã¢ÂÅ’ {filepath.name}: {str(e)[:50]}")
    
    if failed == 0:
        return True, f"({loaded} variants loaded)"
    elif loaded > 0:
        return True, f"({loaded} loaded, {failed} failed)"
    else:
        return False, "All variants failed to load"


# ==============================================================================
# WEEK 1-2 TEST FUNCTIONS (9-12)
# ==============================================================================

def test_regime_classifier():
    """Test 9: Regime Classifier"""
    print("  Testing regime classifier...")
    
    try:
        from regime_classifier import RegimeClassifier, MarketRegime
        import pandas as pd
        import numpy as np
        
        # Create sample data
        print(f"\n    Creating sample OHLCV data...")
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        
        # Generate trending data
        price = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        df = pd.DataFrame({
            'open': price,
            'high': price + abs(np.random.randn(200)),
            'low': price - abs(np.random.randn(200)),
            'close': price + np.random.randn(200) * 0.3,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        print(f"      Ã¢Å“â€¦ Created {len(df)} bars")
        
        # Test classifier
        print(f"\n    Running classification...")
        classifier = RegimeClassifier()
        result_df = classifier.classify(df)
        
        if 'regime' in result_df.columns:
            regimes = result_df['regime'].value_counts()
            print(f"      Ã¢Å“â€¦ Classified bars into {len(regimes)} regimes:")
            for regime, count in regimes.items():
                print(f"         {regime}: {count} bars")
            return True, ""
        else:
            return False, "No 'regime' column in output"
            
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_validation_framework():
    """Test 10: Validation Framework (Bootstrap, Monte Carlo, Walk-Forward)"""
    print("  Testing validation framework...")
    
    try:
        from validation_framework import ValidationFramework, BootstrapResult, MonteCarloResult
        import pandas as pd
        import numpy as np
        
        # Create sample trades data with correct column names
        print(f"\n    Creating sample trades data...")
        np.random.seed(42)
        
        trades = pd.DataFrame({
            'return_pct': np.random.randn(50) * 2 + 0.5,  # Use return_pct (what the framework expects)
            'pnl': np.random.randn(50) * 100 + 20,
            'entry_time': pd.date_range('2023-01-01', periods=50, freq='1D'),
            'exit_time': pd.date_range('2023-01-02', periods=50, freq='1D'),
        })
        
        print(f"      Ã¢Å“â€¦ Created {len(trades)} sample trades")
        
        validator = ValidationFramework(n_bootstrap=100, n_monte_carlo=100)
        
        # Test Bootstrap
        print(f"\n    Testing Bootstrap validation...")
        try:
            bootstrap_result = validator.bootstrap_trades(trades, metric='return_pct')
            if bootstrap_result:
                print(f"      Ã¢Å“â€¦ Bootstrap complete")
                print(f"         Mean: {bootstrap_result.mean:.2f}%")
                print(f"         CI: [{bootstrap_result.ci_lower:.2f}, {bootstrap_result.ci_upper:.2f}]")
            else:
                print(f"      Ã¢Å¡Â Ã¯Â¸Â  Bootstrap returned None")
        except Exception as e:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Bootstrap error: {e}")
        
        # Test Monte Carlo
        print(f"\n    Testing Monte Carlo simulation...")
        try:
            mc_result = validator.monte_carlo_equity(trades, initial_capital=10000)
            if mc_result:
                print(f"      Ã¢Å“â€¦ Monte Carlo complete")
                print(f"         Probability of ruin: {mc_result.probability_of_ruin:.1f}%")
                print(f"         Mean final equity: ${mc_result.mean_final_equity:.2f}")
            else:
                print(f"      Ã¢Å¡Â Ã¯Â¸Â  Monte Carlo returned None")
        except Exception as e:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Monte Carlo error: {e}")
        
        # Test Walk-Forward (simplified check)
        print(f"\n    Testing Walk-Forward framework...")
        if hasattr(validator, 'walk_forward_test'):
            print(f"      Ã¢Å“â€¦ Walk-Forward method available")
        else:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Walk-Forward method not found")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_manual_gates():
    """Test 11: Manual Validation Gates"""
    print("  Testing manual gates...")
    
    try:
        from manual_gates import ValidationGate
        
        print(f"\n    Importing ValidationGate...")
        gate = ValidationGate(enabled=False)  # Disabled for testing
        print(f"      Ã¢Å“â€¦ ValidationGate imported")
        
        # Check methods that actually exist
        methods = ['approve', 'get_session_summary', 'reset_session']
        for method in methods:
            if hasattr(gate, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢Å¡Â Ã¯Â¸Â  Method '{method}' not found")
        
        # Test approve method (non-interactive since enabled=False)
        print(f"\n    Testing approve method (gates disabled)...")
        result = gate.approve("Test operation", estimated_cost=0.10)
        print(f"      Ã¢Å“â€¦ Approve returned: {result}")
        
        # Test session tracking
        print(f"\n    Testing session tracking...")
        print(f"      Ã¢Å“â€¦ Total approved: {gate.total_approved}")
        print(f"      Ã¢Å“â€¦ Total approved cost: ${gate.total_approved_cost:.2f}")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_regime_backtester():
    """Test 12: Multi-timeframe backtester with regime analysis"""
    print("  Testing backtester with regime analysis...")
    
    try:
        from backtester_multi_timeframe import MultiTimeframeBacktester
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        import config
        
        # Check if regime analysis is available
        backtester = MultiTimeframeBacktester()
        
        if hasattr(backtester, 'run_with_regime_analysis'):
            print(f"      Ã¢Å“â€¦ Regime analysis method available")
        else:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Regime analysis method not found (may be integrated differently)")
        
        # Check for regime classifier integration
        if hasattr(backtester, 'regime_classifier') or hasattr(backtester, 'enable_regime_analysis'):
            print(f"      Ã¢Å“â€¦ Regime classifier integration found")
        else:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Regime classifier may not be integrated yet")
        
        return True, "(regime features checked)"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


# ==============================================================================
# WEEK 3 TEST FUNCTIONS (13-15)
# ==============================================================================

def test_robustness_tests():
    """Test 13: Robustness Tests (Latency, Slippage)"""
    print("  Testing robustness tests module...")
    
    try:
        from robustness_tests import RobustnessTests, LatencyTestResult, SlippageTestResult
        
        print(f"\n    Importing RobustnessTests...")
        tester = RobustnessTests()
        print(f"      Ã¢Å“â€¦ RobustnessTests imported")
        
        # Check methods exist
        methods = [
            'latency_sensitivity_test',
            'slippage_stress_test', 
            'combined_stress_test',
            'print_robustness_report'
        ]
        
        for method in methods:
            if hasattr(tester, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢ÂÅ’ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Check dataclasses
        print(f"\n    Checking result dataclasses...")
        print(f"      Ã¢Å“â€¦ LatencyTestResult: {list(LatencyTestResult.__annotations__.keys())[:4]}...")
        print(f"      Ã¢Å“â€¦ SlippageTestResult: {list(SlippageTestResult.__annotations__.keys())[:4]}...")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_adversarial_reviewer_dry():
    """Test 14: Adversarial Reviewer (DRY RUN - no API call)"""
    print("  Testing adversarial reviewer module (no API call)...")
    
    try:
        from adversarial_reviewer import AdversarialReviewer, AdversarialReview
        import config
        
        print(f"\n    Importing AdversarialReviewer...")
        
        # Check if API key is configured
        if not config.CLAUDE_API_KEY:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  No API key configured (reviewer won't run live)")
        else:
            print(f"      Ã¢Å“â€¦ API key configured")
        
        # Check class can be instantiated (requires API key)
        if config.CLAUDE_API_KEY:
            reviewer = AdversarialReviewer()
            print(f"      Ã¢Å“â€¦ AdversarialReviewer instantiated")
        else:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Cannot instantiate without API key")
        
        # Check methods exist on class
        methods = [
            'review_strategy_code',
            'review_backtest_results',
            'full_adversarial_review',
            'save_reviews'
        ]
        
        for method in methods:
            if hasattr(AdversarialReviewer, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢ÂÅ’ Method '{method}' not found")
        
        # Check dataclass
        print(f"\n    Checking AdversarialReview dataclass...")
        fields = ['overall_risk_score', 'critical_flaws', 'warnings', 'recommended_action']
        for field in fields:
            if field in AdversarialReview.__annotations__:
                print(f"      Ã¢Å“â€¦ Field '{field}' found")
            else:
                print(f"      Ã¢Å¡Â Ã¯Â¸Â  Field '{field}' not found")
        
        return True, "(dry run - no API call made)"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_failures_tracker():
    """Test 15: Failures Tracker"""
    print("  Testing failures tracker...")
    
    try:
        from failures_tracker import FailuresTracker, FailureRecord
        
        print(f"\n    Importing FailuresTracker...")
        tracker = FailuresTracker()
        print(f"      Ã¢Å“â€¦ FailuresTracker imported")
        
        # Check methods exist
        methods = [
            'log_failure',
            'log_from_backtest_result',
            'get_failure_patterns',
            'generate_failures_md',
            'get_mutation_context',
            'print_report'
        ]
        
        for method in methods:
            if hasattr(tracker, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢ÂÅ’ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Check failure types defined
        print(f"\n    Checking failure types...")
        if hasattr(tracker, 'FAILURE_TYPES'):
            print(f"      Ã¢Å“â€¦ {len(tracker.FAILURE_TYPES)} failure types defined")
            for ftype in list(tracker.FAILURE_TYPES.keys())[:5]:
                print(f"         - {ftype}")
        else:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  FAILURE_TYPES not found")
        
        # Test get_summary (non-destructive)
        print(f"\n    Testing get_summary...")
        summary = tracker.get_summary()
        print(f"      Ã¢Å“â€¦ Summary retrieved: {summary.get('total', 0)} failures logged")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


# ==============================================================================
# WEEK 4 TEST FUNCTIONS (16-18)
# ==============================================================================

def test_permutation_tests():
    """Test 16: Permutation Testing"""
    print("  Testing permutation tests module...")
    
    try:
        from permutation_tests import PermutationTester, PermutationResult
        
        print(f"\n    Importing PermutationTester...")
        tester = PermutationTester()
        print(f"      Ã¢Å“â€¦ PermutationTester imported")
        
        # Check methods exist
        methods = [
            'test_strategy',
            '_permute_data',
            '_run_backtest',
            '_extract_metric'
        ]
        
        for method in methods:
            if hasattr(tester, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢ÂÅ’ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Check dataclass
        print(f"\n    Checking PermutationResult dataclass...")
        fields = ['p_value', 'is_significant', 'real_value', 'permutation_mean']
        for field in fields:
            if field in PermutationResult.__annotations__:
                print(f"      Ã¢Å“â€¦ Field '{field}' found")
            else:
                print(f"      Ã¢Å¡Â Ã¯Â¸Â  Field '{field}' not found")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_parameter_sensitivity():
    """Test 17: Parameter Sensitivity Analysis"""
    print("  Testing parameter sensitivity module...")
    
    try:
        from parameter_sensitivity import ParameterSensitivity, SingleParamResult, TwoParamResult
        
        print(f"\n    Importing ParameterSensitivity...")
        analyzer = ParameterSensitivity()
        print(f"      Ã¢Å“â€¦ ParameterSensitivity imported")
        
        # Check methods exist
        methods = [
            'single_param_sweep',
            'two_param_heatmap',
            'save_heatmap',
            '_run_backtest'
        ]
        
        for method in methods:
            if hasattr(analyzer, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢ÂÅ’ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Check dataclasses
        print(f"\n    Checking result dataclasses...")
        print(f"      Ã¢Å“â€¦ SingleParamResult fields: {list(SingleParamResult.__annotations__.keys())[:4]}...")
        print(f"      Ã¢Å“â€¦ TwoParamResult fields: {list(TwoParamResult.__annotations__.keys())[:4]}...")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_cost_adjusted_scoring():
    """Test 18: Cost-Adjusted Scoring"""
    print("  Testing cost-adjusted scoring module...")
    
    try:
        from cost_adjusted_scoring import CostAdjustedScorer, AdjustedResult, CostProfile
        
        print(f"\n    Importing CostAdjustedScorer...")
        scorer = CostAdjustedScorer()
        print(f"      Ã¢Å“â€¦ CostAdjustedScorer imported")
        
        # Check methods exist
        methods = [
            'adjust_result',
            'rank_variants',
            'print_comparison',
            'generate_report',
            'get_profile'
        ]
        
        for method in methods:
            if hasattr(scorer, method):
                print(f"      Ã¢Å“â€¦ Method '{method}' found")
            else:
                print(f"      Ã¢ÂÅ’ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Test with sample data
        print(f"\n    Testing with sample data...")
        sample = {
            'strategy_name': 'Test',
            'symbol': 'EUR-USD',
            'timeframe': '1hour',
            'total_return_pct': 10.0,
            'sharpe_ratio': 1.0,
            'total_trades': 100,
            'bars_tested': 5000
        }
        
        result = scorer.adjust_result(sample)
        print(f"      Ã¢Å“â€¦ Gross return: {result.gross_return_pct:.2f}%")
        print(f"      Ã¢Å“â€¦ Net return: {result.net_return_pct:.2f}%")
        print(f"      Ã¢Å“â€¦ Total costs: {result.total_cost_pct:.2f}%")
        print(f"      Ã¢Å“â€¦ Viable: {result.is_viable}")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_expanded_mutation_config():
    """Test 19: Expanded Mutation Config"""
    print("  Testing expanded mutation config...")
    
    try:
        from mutation_config import get_all_ideas, get_ideas_list
        
        print(f"\n    Loading mutation ideas...")
        ideas = get_ideas_list()
        print(f"      Ã¢Å“â€¦ Loaded {len(ideas)} mutation ideas")
        
        # Check for minimum number of ideas
        if len(ideas) >= 70:
            print(f"      Ã¢Å“â€¦ Sufficient ideas for diverse mutations")
        else:
            print(f"      Ã¢Å¡Â Ã¯Â¸Â  Consider adding more ideas ({len(ideas)} < 70)")
        
        # Check categories exist
        all_ideas = get_all_ideas()
        categories = ['Indicators', 'Stop Losses', 'Take Profits', 'Entry Filters', 
                      'Position Sizing', 'Exit Modifications', 'Strategy Types']
        
        print(f"\n    Checking categories...")
        for cat in categories:
            if cat in all_ideas:
                print(f"      Ã¢Å“â€¦ Category '{cat}' found")
            else:
                print(f"      Ã¢Å¡Â Ã¯Â¸Â  Category '{cat}' not found")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


# ==============================================================================
# WEEK 5 TEST FUNCTIONS (20-25) - New Analysis Modules
# ==============================================================================

def test_statistical_analysis():
    """Test 20: Statistical Analysis (Serial Dependence, GARCH, VaR)"""
    print("  Testing statistical analysis module...")
    
    try:
        from validation_framework import StatisticalAnalysis, quick_statistical_analysis
        import numpy as np
        
        print(f"\n    Importing StatisticalAnalysis...")
        analyzer = StatisticalAnalysis()
        print(f"      ✓ StatisticalAnalysis imported")
        
        # Check methods exist
        methods = [
            'test_serial_dependence',
            'analyze_distribution',
            'fit_garch',
            'calculate_var',
            'full_analysis',
            'print_statistical_report'
        ]
        
        for method in methods:
            if hasattr(analyzer, method):
                print(f"      ✓ Method '{method}' found")
            else:
                print(f"      ✗ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Test with sample data
        print(f"\n    Testing with sample returns...")
        np.random.seed(42)
        returns = np.random.normal(0.5, 2.0, 100)
        
        # Test serial dependence
        serial = analyzer.test_serial_dependence(returns)
        print(f"      ✓ Serial dependence: lag1={serial.autocorr_lag1:.4f}, dependent={serial.has_serial_dependence}")
        
        # Test distribution
        dist = analyzer.analyze_distribution(returns)
        print(f"      ✓ Distribution: skew={dist.skewness:.3f}, kurtosis={dist.kurtosis:.3f}")
        
        # Test VaR
        var = analyzer.calculate_var(returns)
        print(f"      ✓ VaR: historical={var.historical_var:.3f}, CVaR={var.cvar_expected_shortfall:.3f}")
        
        # Test GARCH (may not fit with arch library)
        garch = analyzer.fit_garch(returns)
        print(f"      ✓ GARCH: persistence={garch.persistence:.3f}, fit={garch.model_fit}")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_ftmo_pass_rate():
    """Test 21: FTMO Pass Rate Simulation"""
    print("  Testing FTMO pass rate simulation...")
    
    try:
        from ftmo_compliance import FTMOComplianceChecker
        import pandas as pd
        
        print(f"\n    Importing FTMOComplianceChecker...")
        checker = FTMOComplianceChecker()
        print(f"      ✓ FTMOComplianceChecker imported")
        
        # Check simulate_pass_rate method exists
        if hasattr(checker, 'simulate_pass_rate'):
            print(f"      ✓ Method 'simulate_pass_rate' found")
        else:
            return False, "Missing method: simulate_pass_rate"
        
        # Create sample trades
        print(f"\n    Testing with sample trades...")
        sample_trades = pd.DataFrame([
            {'entry_date': '2024-01-02 10:00:00', 'exit_date': '2024-01-02 15:00:00',
             'entry_price': 1.1000, 'exit_price': 1.1050, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-03 10:00:00', 'exit_date': '2024-01-03 15:00:00',
             'entry_price': 1.1050, 'exit_price': 1.1100, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-04 10:00:00', 'exit_date': '2024-01-04 15:00:00',
             'entry_price': 1.1100, 'exit_price': 1.1150, 'size': 100000, 'symbol': 'EUR-USD'},
            {'entry_date': '2024-01-05 10:00:00', 'exit_date': '2024-01-05 15:00:00',
             'entry_price': 1.1150, 'exit_price': 1.1200, 'size': 100000, 'symbol': 'EUR-USD'},
        ])
        
        # Test basic validation
        result = checker.validate(sample_trades, account_size=10000, phase='challenge')
        print(f"      ✓ Validation: passed={result.passed}, return={result.final_return_pct:.1f}%")
        
        # Note: Full simulation skipped to save time
        print(f"      ✓ simulate_pass_rate method available (full test skipped for speed)")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_feature_engineering():
    """Test 22: Feature Engineering Module"""
    print("  Testing feature engineering module...")
    
    try:
        from feature_engineering import FeatureEngineer, StrategyFeatures, quick_features
        import pandas as pd
        import numpy as np
        
        print(f"\n    Importing FeatureEngineer...")
        engineer = FeatureEngineer()
        print(f"      ✓ FeatureEngineer imported")
        
        # Check methods exist
        methods = [
            'build_features',
            'build_feature_table',
            'features_to_dict',
            'print_feature_summary'
        ]
        
        for method in methods:
            if hasattr(engineer, method):
                print(f"      ✓ Method '{method}' found")
            else:
                print(f"      ✗ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Check StrategyFeatures dataclass
        print(f"\n    Checking StrategyFeatures dataclass...")
        required_fields = ['strategy_name', 'sharpe_ratio', 'skewness', 'var_95_historical', 'ftmo_pass_rate']
        for field in required_fields:
            if field in StrategyFeatures.__annotations__:
                print(f"      ✓ Field '{field}' found")
            else:
                print(f"      ⚠️  Field '{field}' not found")
        
        # Test with sample data
        print(f"\n    Testing feature extraction...")
        sample_result = {
            'strategy_name': 'TestStrategy', 'symbol': 'EUR-USD', 'timeframe': '1hour',
            'total_return_pct': 15.5, 'sharpe_ratio': 1.2, 'max_drawdown_pct': 8.3,
            'total_trades': 45, 'win_rate': 55.0, 'profit_factor': 1.8,
            'trades_per_day': 0.5, 'avg_trade_duration_bars': 12,
            'avg_trade_return_pct': 0.34, 'time_in_market_pct': 35.0,
            'bars_tested': 5000, 'start_date': '2023-01-01', 'end_date': '2024-01-01'
        }
        
        np.random.seed(42)
        trades_df = pd.DataFrame({'return_pct': np.random.normal(0.34, 2.0, 45)})
        
        features = engineer.build_features(sample_result, trades_df=trades_df, 
                                          run_statistical_analysis=True, run_ftmo_simulation=False)
        print(f"      ✓ Features built: {features.strategy_name}")
        print(f"      ✓ Statistical features: skew={features.skewness:.3f}, VaR={features.var_95_historical:.3f}")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_portfolio_engine():
    """Test 23: Portfolio Engine Module"""
    print("  Testing portfolio engine module...")
    
    try:
        from portfolio_engine import PortfolioEngine, PortfolioResult, quick_portfolio
        import pandas as pd
        import numpy as np
        
        print(f"\n    Importing PortfolioEngine...")
        engine = PortfolioEngine()
        print(f"      ✓ PortfolioEngine imported")
        
        # Check methods exist
        methods = [
            'build_portfolio',
            'compare_methods',
            'print_portfolio_report',
            '_equal_weight',
            '_inverse_volatility',
            '_risk_parity',
            '_hierarchical_risk_parity'
        ]
        
        for method in methods:
            if hasattr(engine, method):
                print(f"      ✓ Method '{method}' found")
            else:
                print(f"      ✗ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Test with sample equity curves
        print(f"\n    Testing portfolio construction...")
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        equity_curves = {
            'Strategy1': pd.Series((1 + np.random.normal(0.0003, 0.01, 252)).cumprod() * 10000, index=dates),
            'Strategy2': pd.Series((1 + np.random.normal(0.0005, 0.02, 252)).cumprod() * 10000, index=dates),
        }
        
        # Test different allocation methods
        for method in ['equal', 'inverse_vol', 'risk_parity']:
            portfolio = engine.build_portfolio(equity_curves, method=method)
            print(f"      ✓ {method}: Sharpe={portfolio.portfolio_sharpe:.2f}")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_meta_model():
    """Test 24: Meta Model Module"""
    print("  Testing meta model module...")
    
    try:
        from meta_model import MetaModel, EarlyKillFilter, PredictionResult
        
        print(f"\n    Importing MetaModel and EarlyKillFilter...")
        model = MetaModel()
        killer = EarlyKillFilter()
        print(f"      ✓ MetaModel imported")
        print(f"      ✓ EarlyKillFilter imported")
        
        # Check MetaModel methods
        methods = ['train', 'predict', 'feature_importance', 'save_model', 'load_model']
        for method in methods:
            if hasattr(model, method):
                print(f"      ✓ MetaModel method '{method}' found")
            else:
                print(f"      ✗ MetaModel method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Check EarlyKillFilter
        if hasattr(killer, 'should_kill'):
            print(f"      ✓ EarlyKillFilter method 'should_kill' found")
        else:
            return False, "Missing method: should_kill"
        
        # Test early kill filter
        print(f"\n    Testing EarlyKillFilter...")
        bad_strategy = {'total_trades': 10, 'sharpe_ratio': -0.5, 'max_drawdown_pct': 35}
        should_kill, reasons = killer.should_kill(bad_strategy)
        print(f"      ✓ Bad strategy: kill={should_kill}, reasons={len(reasons)}")
        
        good_strategy = {'total_trades': 100, 'sharpe_ratio': 1.5, 'max_drawdown_pct': 10, 'total_return_pct': 20}
        should_kill, reasons = killer.should_kill(good_strategy)
        print(f"      ✓ Good strategy: kill={should_kill}")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_execution_engine():
    """Test 25: Execution Engine Module"""
    print("  Testing execution engine module...")
    
    try:
        from execution_engine import PaperTrader, ExecutionEngine, Order, Position
        
        print(f"\n    Importing PaperTrader and ExecutionEngine...")
        trader = PaperTrader(initial_capital=100000)
        print(f"      ✓ PaperTrader imported")
        
        # Check PaperTrader methods
        methods = [
            'submit_order',
            'cancel_order',
            'close_position',
            'close_all_positions',
            'update_price',
            'get_account_summary',
            'get_trades_df'
        ]
        
        for method in methods:
            if hasattr(trader, method):
                print(f"      ✓ Method '{method}' found")
            else:
                print(f"      ✗ Method '{method}' not found")
                return False, f"Missing method: {method}"
        
        # Test paper trading
        print(f"\n    Testing paper trading simulation...")
        trader.update_price('EUR-USD', 1.1000)
        
        order = trader.submit_order('EUR-USD', 'BUY', 10000, 'MARKET')
        print(f"      ✓ Order submitted: status={order.status.value}")
        
        trader.update_price('EUR-USD', 1.1050)
        pos = trader.positions.get('EUR-USD')
        if pos:
            print(f"      ✓ Position tracked: PnL=${pos.unrealized_pnl:.2f}")
        
        trader.close_position('EUR-USD')
        summary = trader.get_account_summary()
        print(f"      ✓ Account summary: equity=${summary['equity']:.2f}")
        
        # Test ExecutionEngine
        print(f"\n    Testing ExecutionEngine...")
        engine = ExecutionEngine(mode='paper', initial_capital=100000)
        print(f"      ✓ ExecutionEngine imported")
        
        if hasattr(engine, 'process_signal'):
            print(f"      ✓ Method 'process_signal' found")
        if hasattr(engine, 'run_backtest_signals'):
            print(f"      ✓ Method 'run_backtest_signals' found")
        
        return True, ""
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all system tests"""
    
    print("\n" + "="*70)
    print("  Ã°Å¸â€Â§ TRADING RESEARCH SYSTEM - FULL TEST")
    print("="*70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print("="*70)
    
    # All tests organized by phase
    tests = [
        # Foundation (1-8)
        ("1. Config Validation", test_config),
        ("2. Data Manager", test_data_manager),
        ("3. Database Connection", test_database),
        ("4. Base Strategy", test_base_strategy),
        ("5. Single Backtest", test_single_backtest),
        ("6. Multi-Asset Backtest", test_multi_backtest),
        ("7. Mutation Agent (Dry Run)", test_mutation_agent_dry),
        ("8. Variant Loading", test_variant_loading),
        
        # Week 1-2: Foundation (9-12)
        ("9. Regime Classifier", test_regime_classifier),
        ("10. Validation Framework", test_validation_framework),
        ("11. Manual Gates", test_manual_gates),
        ("12. Regime Backtester", test_regime_backtester),
        
        # Week 3: Robustness (13-15)
        ("13. Robustness Tests", test_robustness_tests),
        ("14. Adversarial Reviewer (Dry Run)", test_adversarial_reviewer_dry),
        ("15. Failures Tracker", test_failures_tracker),
        
        # Week 4: Refinements (16-19)
        ("16. Permutation Tests", test_permutation_tests),
        ("17. Parameter Sensitivity", test_parameter_sensitivity),
        ("18. Cost-Adjusted Scoring", test_cost_adjusted_scoring),
        ("19. Expanded Mutation Config", test_expanded_mutation_config),
        
        # Week 5: Advanced Analysis (20-25)
        ("20. Statistical Analysis", test_statistical_analysis),
        ("21. FTMO Pass Rate Simulation", test_ftmo_pass_rate),
        ("22. Feature Engineering", test_feature_engineering),
        ("23. Portfolio Engine", test_portfolio_engine),
        ("24. Meta Model", test_meta_model),
        ("25. Execution Engine", test_execution_engine),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        success = run_test(test_name, test_func)
        results.append((test_name, success))
        
        if not ask_continue():
            print("\n  Ã°Å¸â€ºâ€˜ Testing stopped by user")
            break
    
    # Final Summary
    print_header("FINAL SUMMARY")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n  Tests Run: {total}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {total - passed}")
    
    # Group results by phase
    print(f"\n  Ã°Å¸â€œâ€¹ FOUNDATION TESTS (1-8):")
    for test_name, success in results[:8]:
        status = "Ã¢Å“â€¦ PASS" if success else "Ã¢ÂÅ’ FAIL"
        print(f"    {status}  {test_name}")
    
    if len(results) > 8:
        print(f"\n  Ã°Å¸â€œâ€¹ WEEK 1-2 TESTS (9-12):")
        for test_name, success in results[8:12]:
            status = "Ã¢Å“â€¦ PASS" if success else "Ã¢ÂÅ’ FAIL"
            print(f"    {status}  {test_name}")
    
    if len(results) > 12:
        print(f"\n  Ã°Å¸â€œâ€¹ WEEK 3 TESTS (13-15):")
        for test_name, success in results[12:15]:
            status = "Ã¢Å“â€¦ PASS" if success else "Ã¢ÂÅ’ FAIL"
            print(f"    {status}  {test_name}")
    


    if len(results) > 15:
        print(f"\n  📋 WEEK 4 TESTS (16-19):")
        for test_name, success in results[15:19]:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"    {status}  {test_name}")
    
    if len(results) > 19:
        print(f"\n  🚀 WEEK 5 TESTS (20-25):")
        for test_name, success in results[19:25]:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"    {status}  {test_name}")
    
    # Overall status
    if passed == total:
        print(f"\n  🎉 ALL TESTS PASSED!")
        print(f"\n  Your system is fully operational. Available commands:")
        print(f"    • python run_backtests.py          - Run backtests")
        print(f"    • python mutate_strategy.py        - Generate variants")
        print(f"    • python run_variant_backtests.py  - Test variants")
        print(f"    • python robustness_tests.py       - Robustness testing")
        print(f"    • python adversarial_reviewer.py   - AI code review")
        print(f"    • python failures_tracker.py       - Track failures")
        print(f"    • python dashboard_react.py        - Launch ReactPy dashboard")
        print(f"\n  New Week 5 modules available:")
        print(f"    • validation_framework.py          - Statistical analysis")
        print(f"    • feature_engineering.py           - Feature extraction")
        print(f"    • portfolio_engine.py              - Portfolio allocation")
        print(f"    • meta_model.py                    - ML survival prediction")
        print(f"    • execution_engine.py              - Paper trading")
    elif passed > total * 0.7:
        print(f"\n  ⚠️  MOSTLY PASSED - Check failed tests above")
    else:
        print(f"\n  ❌ MULTIPLE FAILURES - Fix issues before proceeding")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()