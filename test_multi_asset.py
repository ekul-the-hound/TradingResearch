# test_multi_asset.py
# Your first AI-assisted multi-asset backtest!

from backtester import StrategyBacktester
from strategies.simple_strategy import SimpleMovingAverageCrossover
import config

print("\n" + "="*70)
print("PHASE 2: AI-ASSISTED MULTI-ASSET BACKTEST")
print("="*70)
print("\nThis will:")
print("  [OK] Test your strategy across 10 major stocks")
print("  [OK] Save all results to a database")
print("  [OK] Ask Claude AI to analyze the results")
print("  [OK] Get insights and suggestions")
print("="*70)

# Create the enhanced backtester
backtester = StrategyBacktester()

# Run multi-asset backtest on top 10 stocks
print("\n[LAUNCH] Starting multi-asset backtest on major stocks...")
print(f"   Testing: {', '.join(config.STOCK_WATCHLIST)}")

results = backtester.run_multi_asset_backtest(
    strategy_class=SimpleMovingAverageCrossover,
    symbols=config.STOCK_WATCHLIST,
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_cash=10000,
    commission=0.001
)

print("\n" + "="*70)
print("[OK] PHASE 2 COMPLETE!")
print("="*70)
print(f"\nResults saved to database: {config.DATABASE_PATH}")
print("Claude's analysis is shown above ^")
print("\nNext steps:")
print("  -> Review Claude's insights")
print("  -> Try different date ranges")
print("  -> Test on crypto or forex")
print("  -> Phase 3: Build component library")
print("  -> Phase 4: Add the Mutation Agent")
print("="*70 + "\n")