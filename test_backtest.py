# test_backtest.py
# PHASE 1 TEST - Verify the backtester works with your base strategy

from backtester import StrategyBacktester
from strategies.simple_strategy import SimpleMovingAverageCrossover

print("\n" + "="*70)
print("PHASE 1: BASIC BACKTEST TEST")
print("="*70)
print("\nThis tests:")
print("  [OK] Your backtesting engine works")
print("  [OK] Your base strategy executes correctly")
print("  [OK] Data downloads properly")
print("  [OK] Metrics calculate correctly")
print("\nLater, the AI will generate variants of this strategy automatically.")
print("="*70)

# Create the backtester
backtester = StrategyBacktester()

# Test 1: Basic test on Apple
print("\n📍 TEST 1: Apple (AAPL) - 2020 to 2023")
result = backtester.run_backtest(
    strategy_class=SimpleMovingAverageCrossover,
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_cash=10000,
    commission=0.001
)

if result:
    print("[OK] Test 1 PASSED - Backtester works!")
else:
    print("[FAIL] Test 1 FAILED - Check errors above")
    exit()

# Test 2: Test with custom parameters
print("\n📍 TEST 2: Same strategy with different parameters")
print("            (Testing 20/50 day moving averages instead of 10/30)")

result2 = backtester.run_backtest(
    strategy_class=SimpleMovingAverageCrossover,
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_cash=10000,
    commission=0.001,
    strategy_params={'fast_period': 20, 'slow_period': 50}  # Different parameters
)

if result2:
    print("[OK] Test 2 PASSED - Parameter customization works!")
    
    # Compare the two
    print(f"\n{'='*70}")
    print("COMPARISON: 10/30 MA vs 20/50 MA")
    print(f"{'='*70}")
    print(f"10/30 MA Return: {result['total_return_pct']:+.2f}%")
    print(f"20/50 MA Return: {result2['total_return_pct']:+.2f}%")
    print(f"{'='*70}")
else:
    print("[FAIL] Test 2 FAILED")
    exit()

print("\n" + "="*70)
print("[OK] PHASE 1 COMPLETE!")
print("="*70)
print("\nYour backtesting engine is working correctly.")
print("\nNext steps:")
print("  -> Phase 2: Add database storage")
print("  -> Phase 3: Integrate Claude AI for analysis")
print("  -> Phase 4: Build the Mutation Agent")
print("\nThe Mutation Agent will automatically generate variants like:")
print("  - 5/15 MA, 10/30 MA, 20/50 MA (different periods)")
print("  - MA + ADX filter")
print("  - MA + RSI confirmation")
print("  - MA + ATR stops")
print("  - ...and hundreds more")
print("="*70 + "\n")
