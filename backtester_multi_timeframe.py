# ==============================================================================
# backtester_multi_timeframe.py
# ==============================================================================
# Multi-asset, multi-timeframe backtesting engine
#
# UPDATED: Combined with regime analysis, trade extraction, and manual gates
#
# Features:
# - Multi-asset, multi-timeframe batch testing
# - Regime-segmented performance breakdown (BULL/BEAR/RANGING/etc.)
# - Trade-level extraction for validation framework integration
# - Manual validation gates (optional)
# - Enhanced metrics per regime
#
# Usage:
#     from backtester_multi_timeframe import MultiTimeframeBacktester
#     
#     backtester = MultiTimeframeBacktester()
#     
#     # Standard batch run (existing behavior)
#     results = backtester.run_multi_asset_multi_timeframe(
#         strategy_class=MyStrategy,
#         assets=['EUR-USD', 'BTC-USD'],
#         timeframes=['1hour', '4hour']
#     )
#     
#     # With regime analysis (new)
#     results = backtester.run_multi_asset_multi_timeframe(
#         ...,
#         regime_analysis=True
#     )
#     
#     # Single backtest with full regime report
#     result = backtester.run_with_regime_analysis(
#         strategy_class=MyStrategy,
#         symbol='EUR-USD',
#         timeframe='1hour'
#     )
#
# ==============================================================================

import backtrader as bt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings

import config
from database import ResultsDatabase
from data_manager import DataManager

# Try to import regime classifier (optional dependency)
try:
    from regime_classifier import RegimeClassifier, MarketRegime
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    print("⚠️  regime_classifier.py not found - regime analysis disabled")
    print("   Copy regime_classifier.py to your project to enable")


# ==============================================================================
# TRADE TRACKER ANALYZER
# ==============================================================================

class TradeTracker(bt.Analyzer):
    """
    Custom analyzer to track individual trades.
    
    Captures trade-level data needed for:
    - Bootstrap validation
    - Monte Carlo simulation
    - Regime-segmented analysis
    """
    
    def __init__(self):
        self.trades = []
    
    def notify_trade(self, trade):
        if trade.isclosed:
            # Calculate trade return
            pnl = trade.pnl
            pnl_pct = (trade.pnl / abs(trade.price * trade.size)) * 100 if trade.price else 0
            
            trade_record = {
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'entry_price': trade.price,
                'exit_price': trade.price + (trade.pnl / trade.size) if trade.size else 0,
                'size': trade.size,
                'pnl': pnl,
                'return_pct': pnl_pct,
                'duration_bars': trade.barlen,
                'is_long': trade.size > 0,
            }
            
            self.trades.append(trade_record)
    
    def get_analysis(self):
        return {'trades': self.trades}


# ==============================================================================
# MAIN BACKTESTER CLASS
# ==============================================================================

class MultiTimeframeBacktester:
    """
    Tests strategies across multiple assets and timeframes.
    
    Now includes:
    - Regime-segmented analysis (optional)
    - Trade-level extraction for validation
    - Manual validation gates (optional)
    """
    
    def __init__(self, enable_gates: bool = False):
        """
        Args:
            enable_gates: If True, pause for manual approval before batch tests
        """
        self.db = ResultsDatabase()
        self.data_manager = DataManager()
        self.results = []
        
        # Manual gates
        self.enable_gates = enable_gates
        
        # Regime classifier (if available)
        if REGIME_AVAILABLE:
            self.regime_classifier = RegimeClassifier()
        else:
            self.regime_classifier = None
        
        # Tracking
        self.tests_run = 0
    
    # =========================================================================
    # MANUAL VALIDATION GATES
    # =========================================================================
    
    def _manual_gate(self, description: str, estimated_cost: float = 0) -> bool:
        """
        Manual validation gate - pause for user approval.
        
        Args:
            description: What will be run if approved
            estimated_cost: Estimated API cost (if any)
        
        Returns:
            True if approved, False if rejected
        """
        if not self.enable_gates:
            return True
        
        print("\n" + "="*70)
        print("🚦 MANUAL VALIDATION GATE")
        print("="*70)
        print(f"\nProposed action: {description}")
        
        if estimated_cost > 0:
            print(f"Estimated cost: ${estimated_cost:.2f}")
        
        print("\nOptions:")
        print("  [Y] Proceed")
        print("  [N] Skip this step")
        print("  [A] Approve all remaining (disable gates)")
        print("  [Q] Quit")
        
        while True:
            try:
                response = input("\nYour choice (Y/N/A/Q): ").strip().upper()
            except EOFError:
                return True  # Non-interactive mode
            
            if response == 'Y':
                return True
            elif response == 'N':
                print("⏭️  Skipped")
                return False
            elif response == 'A':
                print("✅ Gates disabled for this session")
                self.enable_gates = False
                return True
            elif response == 'Q':
                print("🛑 Aborted by user")
                raise KeyboardInterrupt("User quit at validation gate")
            else:
                print("Invalid choice. Please enter Y, N, A, or Q")
    
    # =========================================================================
    # SINGLE BACKTEST
    # =========================================================================
    
    def run_single_backtest(
        self, 
        strategy_class, 
        symbol, 
        timeframe, 
        initial_cash=None,
        commission=None,
        strategy_params=None,
        extract_trades: bool = False
    ):
        """
        Run a single backtest on one asset/timeframe combination.
        
        Args:
            strategy_class: Backtrader strategy class
            symbol: Asset symbol
            timeframe: Timeframe string
            initial_cash: Starting capital (default from config)
            commission: Commission rate (default from config)
            strategy_params: Strategy parameters dict
            extract_trades: If True, include individual trade list in results
        
        Returns:
            Dictionary with results, or None if failed
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if commission is None:
            commission = config.DEFAULT_COMMISSION
        
        # Get data from cache
        try:
            data = self.data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                max_bars=config.CANDLE_LIMITS.get(timeframe, 1000)
            )
        except Exception as e:
            print(f"   ❌ Data error: {e}")
            return None
        
        if data is None or data.empty:
            print(f"   ⏭️  Skipped (no data)")
            return None
        
        if len(data) < 50:  # Minimum bars for meaningful backtest
            print(f"   ⏭️  Skipped (only {len(data)} bars, need 50+)")
            return None
        
        # Clean up the data - rename columns to lowercase
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only OHLCV columns
        available_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in data.columns]
        data = data[available_cols]
        
        # Normalize timezone explicitly using UTC conversion
        if data.index.tz is not None:
            data.index = data.index.tz_convert("UTC").tz_localize(None)
        
        # Create backtrader engine
        cerebro = bt.Cerebro()
        
        # Prepare data feed
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume' if 'volume' in data.columns else None,
            openinterest=-1
        )
        cerebro.adddata(data_feed)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Configure broker
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Add trade tracker if extracting trades
        if extract_trades:
            cerebro.addanalyzer(TradeTracker, _name='trade_tracker')
        
        # Run backtest
        starting_value = cerebro.broker.getvalue()
        
        try:
            results = cerebro.run()
        except Exception as e:
            print(f"   ❌ Backtest failed: {e}")
            return None
        
        ending_value = cerebro.broker.getvalue()
        
        # Extract results
        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        
        # Calculate metrics
        total_return = ((ending_value - starting_value) / starting_value) * 100
        
        # Safe extraction of trade stats
        total_trades = 0
        try:
            total_trades = trades.total.closed or 0
        except (KeyError, AttributeError):
            total_trades = 0
        
        # Win rate
        win_rate = None
        if total_trades > 0:
            try:
                wins = trades.won.total or 0
                win_rate = (wins / total_trades) * 100
            except (KeyError, AttributeError):
                win_rate = None
        
        # Profit factor
        profit_factor = None
        try:
            gross_profit = trades.won.pnl.total or 0
            gross_loss = abs(trades.lost.pnl.total or 0)
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = float('inf')
        except (KeyError, AttributeError):
            profit_factor = None
        
        # Build result
        result = {
            'strategy_name': strategy_class.__name__,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'bars_tested': len(data),
            'starting_value': starting_value,
            'ending_value': ending_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe.get('sharperatio'),
            'max_drawdown_pct': drawdown.max.drawdown if hasattr(drawdown, 'max') else 0,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else None,
            'strategy_params': strategy_params or {}
        }
        
        # Extract individual trades if requested
        if extract_trades:
            trade_tracker = strat.analyzers.trade_tracker.get_analysis()
            result['trades'] = trade_tracker['trades']
            result['_data'] = data  # Keep data for regime analysis (will be removed later)
        
        # Print summary
        sharpe_str = f"{result['sharpe_ratio']:.2f}" if result['sharpe_ratio'] else "N/A"
        print(f"   ✔ {symbol:12} {timeframe:6} | Return: {total_return:+6.2f}% | Sharpe: {sharpe_str:>6} | Trades: {total_trades:3} | Bars: {len(data):4}")
        
        self.tests_run += 1
        
        return result
    
    # =========================================================================
    # REGIME ANALYSIS
    # =========================================================================
    
    def run_with_regime_analysis(
        self,
        strategy_class,
        symbol: str,
        timeframe: str,
        initial_cash: float = None,
        commission: float = None,
        strategy_params: Dict = None,
        save_to_db: bool = True
    ) -> Optional[Dict]:
        """
        Run backtest with full regime-segmented analysis.
        
        Returns:
            Dictionary with overall results + regime breakdown
        """
        
        if not REGIME_AVAILABLE:
            print("⚠️  Regime analysis not available - running standard backtest")
            return self.run_single_backtest(
                strategy_class, symbol, timeframe,
                initial_cash, commission, strategy_params
            )
        
        # Run base backtest with trade extraction
        result = self.run_single_backtest(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=timeframe,
            initial_cash=initial_cash,
            commission=commission,
            strategy_params=strategy_params,
            extract_trades=True
        )
        
        if result is None:
            return None
        
        # Classify regimes in the data
        data = result.pop('_data', None)
        if data is None:
            print("⚠️  No data for regime classification")
            return result
        
        data_with_regimes = self.regime_classifier.classify(data)
        
        # Map trades to regimes
        trades = result.get('trades', [])
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Assign regime to each trade based on entry date
            trade_regimes = []
            for _, trade in trades_df.iterrows():
                entry_date = trade['entry_date']
                
                # Find regime at entry
                try:
                    idx = data_with_regimes.index.get_indexer([entry_date], method='nearest')[0]
                    regime = data_with_regimes.iloc[idx]['regime']
                except:
                    regime = 'UNKNOWN'
                
                trade_regimes.append(regime)
            
            trades_df['regime'] = trade_regimes
            
            # Calculate regime-segmented statistics
            regime_stats = {}
            
            for regime in MarketRegime:
                regime_trades = trades_df[trades_df['regime'] == regime.value]
                
                if len(regime_trades) > 0:
                    regime_stats[regime.value] = {
                        'n_trades': len(regime_trades),
                        'total_return': regime_trades['return_pct'].sum(),
                        'avg_return': regime_trades['return_pct'].mean(),
                        'win_rate': (regime_trades['return_pct'] > 0).mean() * 100,
                        'avg_duration': regime_trades['duration_bars'].mean(),
                        'best_trade': regime_trades['return_pct'].max(),
                        'worst_trade': regime_trades['return_pct'].min(),
                    }
            
            result['regime_stats'] = regime_stats
            result['trades_df'] = trades_df
        else:
            result['regime_stats'] = {}
            result['trades_df'] = pd.DataFrame()
        
        # Get regime summary for the data period
        result['regime_summary'] = self.regime_classifier.get_regime_summary(data_with_regimes)
        
        # Save to database
        if save_to_db:
            self.db.save_backtest(result)
        
        return result
    
    def print_regime_report(self, result: Dict):
        """Print formatted regime-segmented report"""
        
        print("\n" + "="*70)
        print(f"REGIME-SEGMENTED BACKTEST REPORT")
        print("="*70)
        print(f"Symbol:    {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Period:    {result['start_date']} to {result['end_date']}")
        print(f"Bars:      {result['bars_tested']}")
        print("="*70)
        
        # Overall performance
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Return:    {result['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio:    {result['sharpe_ratio']:.2f}" if result['sharpe_ratio'] else "  Sharpe Ratio:    N/A")
        print(f"  Max Drawdown:    {result['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades:    {result['total_trades']}")
        print(f"  Win Rate:        {result['win_rate']:.1f}%" if result['win_rate'] else "  Win Rate:        N/A")
        
        # Regime distribution
        if 'regime_summary' in result and result['regime_summary']:
            print("\n📈 MARKET REGIME DISTRIBUTION:")
            print(f"  {'Regime':<12} {'% of Time':>10} {'Avg Return':>12}")
            print("  " + "-"*40)
            
            for regime, stats in result['regime_summary'].items():
                avg_ret = f"{stats['avg_return']:.3f}%" if stats['avg_return'] else "N/A"
                print(f"  {regime:<12} {stats['pct_of_total']:>9.1f}% {avg_ret:>12}")
        
        # Performance by regime
        if 'regime_stats' in result and result['regime_stats']:
            print("\n🎯 STRATEGY PERFORMANCE BY REGIME:")
            print(f"  {'Regime':<12} {'Trades':>8} {'Avg Ret':>10} {'Win Rate':>10} {'Total':>10}")
            print("  " + "-"*55)
            
            for regime, stats in sorted(result['regime_stats'].items(), 
                                        key=lambda x: x[1]['total_return'], reverse=True):
                print(f"  {regime:<12} {stats['n_trades']:>8} {stats['avg_return']:>9.2f}% "
                      f"{stats['win_rate']:>9.1f}% {stats['total_return']:>9.2f}%")
            
            # Identify problematic regimes
            print("\n💡 INSIGHTS:")
            
            problem_regimes = []
            strong_regimes = []
            
            for regime, stats in result['regime_stats'].items():
                if stats['avg_return'] < -0.5:
                    problem_regimes.append((regime, stats['avg_return']))
                elif stats['avg_return'] > 0.5 and stats['n_trades'] >= 5:
                    strong_regimes.append((regime, stats['avg_return']))
            
            if strong_regimes:
                print("  ✅ Strategy works well in: " + 
                      ", ".join([f"{r} ({v:+.2f}%/trade)" for r, v in strong_regimes]))
            
            if problem_regimes:
                print("  ⚠️  Strategy struggles in: " + 
                      ", ".join([f"{r} ({v:+.2f}%/trade)" for r, v in problem_regimes]))
                print("     → Consider adding regime filter to avoid these conditions")
            
            if not problem_regimes and not strong_regimes:
                print("  ℹ️  No clear regime preference detected")
        
        print("\n" + "="*70)
    
    # =========================================================================
    # MULTI-ASSET MULTI-TIMEFRAME BATCH
    # =========================================================================
    
    def run_multi_asset_multi_timeframe(
        self, 
        strategy_class, 
        assets, 
        timeframes,
        initial_cash=None,
        commission=None,
        strategy_params=None,
        save_to_db=True,
        regime_analysis=False,
        extract_trades=False
    ):
        """
        Run backtests across multiple assets and timeframes.
        
        Args:
            strategy_class: Backtrader strategy class
            assets: List of asset symbols
            timeframes: List of timeframe strings
            initial_cash: Starting capital
            commission: Commission rate
            strategy_params: Strategy parameters dict
            save_to_db: Save results to database
            regime_analysis: If True, include regime-segmented analysis
            extract_trades: If True, include individual trade lists
        
        Returns:
            List of result dictionaries
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if commission is None:
            commission = config.DEFAULT_COMMISSION
        
        total_tests = len(assets) * len(timeframes)
        current_test = 0
        results = []
        
        # Manual gate before starting
        if self.enable_gates:
            if not self._manual_gate(
                f"Run {total_tests} backtests across {len(assets)} assets and {len(timeframes)} timeframes"
            ):
                return []
        
        print(f"\n{'='*80}")
        print(f"MULTI-ASSET MULTI-TIMEFRAME BACKTEST")
        print(f"{'='*80}")
        print(f"Strategy:        {strategy_class.__name__}")
        print(f"Assets:          {len(assets)}")
        print(f"Timeframes:      {len(timeframes)}")
        print(f"Total Tests:     {total_tests}")
        print(f"Regime Analysis: {'Yes' if regime_analysis else 'No'}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Group assets by type for organized output
        asset_groups = {
            'Forex': [a for a in assets if a in config.FOREX_WATCHLIST],
            'Indices': [a for a in assets if a in config.INDEX_WATCHLIST],
            'Commodities': [a for a in assets if a in config.COMMODITY_WATCHLIST],
            'Crypto': [a for a in assets if a in config.CRYPTO_WATCHLIST]
        }
        
        # Test each asset group
        for group_name, group_assets in asset_groups.items():
            if not group_assets:
                continue
            
            print(f"\n{'─'*80}")
            print(f"Testing {group_name} ({len(group_assets)} assets)")
            print(f"{'─'*80}")
            
            for symbol in group_assets:
                print(f"\n📊 {symbol}")
                
                for timeframe in timeframes:
                    current_test += 1
                    progress = (current_test / total_tests) * 100
                    
                    print(f"   [{current_test:3}/{total_tests}] ({progress:5.1f}%) Testing {timeframe:6}...", end=" ")
                    
                    # Run appropriate backtest type
                    if regime_analysis and REGIME_AVAILABLE:
                        result = self.run_with_regime_analysis(
                            strategy_class=strategy_class,
                            symbol=symbol,
                            timeframe=timeframe,
                            initial_cash=initial_cash,
                            commission=commission,
                            strategy_params=strategy_params,
                            save_to_db=save_to_db
                        )
                    else:
                        result = self.run_single_backtest(
                            strategy_class=strategy_class,
                            symbol=symbol,
                            timeframe=timeframe,
                            initial_cash=initial_cash,
                            commission=commission,
                            strategy_params=strategy_params,
                            extract_trades=extract_trades
                        )
                        
                        if result and save_to_db:
                            self.db.save_backtest(result)
                    
                    if result:
                        results.append(result)
        
        # Summary
        elapsed_time = time.time() - start_time
        successful = len(results)
        failed = total_tests - successful
        
        print(f"\n{'='*80}")
        print(f"BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Successful:   {successful}/{total_tests} ({successful/total_tests*100:.1f}%)")
        print(f"Failed:       {failed}/{total_tests}")
        print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
        
        if save_to_db:
            print(f"Results saved to: {config.DATABASE_PATH}")
        
        print(f"{'='*80}\n")
        
        self.results = results
        return results
    
    # =========================================================================
    # TRADE EXTRACTION FOR VALIDATION
    # =========================================================================
    
    def get_trades_for_validation(self, results: List[Dict] = None) -> pd.DataFrame:
        """
        Extract all trades from batch results for validation framework.
        
        Args:
            results: List of result dicts (default: self.results)
        
        Returns:
            DataFrame with all trades suitable for bootstrap/monte carlo
        """
        
        if results is None:
            results = self.results
        
        all_trades = []
        
        for result in results:
            # Check for trades_df (from regime analysis)
            if 'trades_df' in result and len(result['trades_df']) > 0:
                trades_df = result['trades_df'].copy()
                trades_df['symbol'] = result['symbol']
                trades_df['timeframe'] = result['timeframe']
                all_trades.append(trades_df)
            # Check for trades list (from extract_trades=True)
            elif 'trades' in result and result['trades']:
                trades_df = pd.DataFrame(result['trades'])
                trades_df['symbol'] = result['symbol']
                trades_df['timeframe'] = result['timeframe']
                all_trades.append(trades_df)
        
        if all_trades:
            return pd.concat(all_trades, ignore_index=True)
        else:
            return pd.DataFrame()
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    def get_summary_stats(self, results=None):
        """
        Calculate summary statistics across all results
        """
        if results is None:
            results = self.results
        
        if not results:
            return None
        
        # Overall stats
        avg_return = sum(r['total_return_pct'] for r in results) / len(results)
        sharpe_results = [r for r in results if r['sharpe_ratio'] is not None]
        avg_sharpe = sum(r['sharpe_ratio'] for r in sharpe_results) / len(sharpe_results) if sharpe_results else None
        avg_dd = sum(r['max_drawdown_pct'] for r in results) / len(results)
        avg_trades = sum(r['total_trades'] for r in results) / len(results)
        
        # Best/worst
        best = max(results, key=lambda x: x['total_return_pct'])
        worst = min(results, key=lambda x: x['total_return_pct'])
        
        # By timeframe
        timeframe_stats = {}
        for tf in set(r['timeframe'] for r in results):
            tf_results = [r for r in results if r['timeframe'] == tf]
            if tf_results:
                timeframe_stats[tf] = {
                    'count': len(tf_results),
                    'avg_return': sum(r['total_return_pct'] for r in tf_results) / len(tf_results),
                    'positive_count': len([r for r in tf_results if r['total_return_pct'] > 0])
                }
        
        return {
            'total_tests': len(results),
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_dd,
            'avg_trades': avg_trades,
            'best_performer': best,
            'worst_performer': worst,
            'timeframe_stats': timeframe_stats
        }
    
    def get_regime_summary_stats(self, results=None) -> Dict:
        """
        Get aggregated regime statistics across all results.
        
        Returns:
            Dictionary with regime-level performance aggregates
        """
        
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        # Collect all regime stats
        regime_totals = {}
        
        for result in results:
            if 'regime_stats' not in result:
                continue
            
            for regime, stats in result['regime_stats'].items():
                if regime not in regime_totals:
                    regime_totals[regime] = {
                        'total_trades': 0,
                        'returns': [],
                        'win_rates': []
                    }
                
                regime_totals[regime]['total_trades'] += stats['n_trades']
                regime_totals[regime]['returns'].append(stats['avg_return'])
                regime_totals[regime]['win_rates'].append(stats['win_rate'])
        
        # Calculate averages
        regime_summary = {}
        for regime, totals in regime_totals.items():
            if totals['returns']:
                regime_summary[regime] = {
                    'total_trades': totals['total_trades'],
                    'avg_return': np.mean(totals['returns']),
                    'avg_win_rate': np.mean(totals['win_rates']),
                }
        
        return regime_summary


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_regime_backtest(
    strategy_class,
    symbol: str,
    timeframe: str,
    **kwargs
) -> Optional[Dict]:
    """Quick function to run regime-segmented backtest"""
    backtester = MultiTimeframeBacktester(enable_gates=False)
    return backtester.run_with_regime_analysis(
        strategy_class=strategy_class,
        symbol=symbol,
        timeframe=timeframe,
        **kwargs
    )


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MULTI-TIMEFRAME BACKTESTER TEST")
    print("="*70)
    
    # Try to import a test strategy
    try:
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        strategy_class = SimpleMovingAverageCrossover
        print("✅ Loaded SimpleMovingAverageCrossover strategy")
    except ImportError:
        print("⚠️  Could not import test strategy, creating simple test strategy...")
        
        class SimpleTestStrategy(bt.Strategy):
            params = (('fast', 10), ('slow', 30))
            
            def __init__(self):
                self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast)
                self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow)
                self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
            
            def next(self):
                if not self.position:
                    if self.crossover > 0:
                        self.buy()
                else:
                    if self.crossover < 0:
                        self.sell()
        
        strategy_class = SimpleTestStrategy
    
    # Initialize backtester
    backtester = MultiTimeframeBacktester(enable_gates=False)
    
    print(f"\nRegime analysis available: {REGIME_AVAILABLE}")
    
    # Test single backtest with regime analysis
    if REGIME_AVAILABLE:
        print("\n" + "-"*70)
        print("Testing single backtest with regime analysis...")
        print("-"*70)
        
        result = backtester.run_with_regime_analysis(
            strategy_class=strategy_class,
            symbol='EUR-USD',
            timeframe='1hour',
            save_to_db=False
        )
        
        if result:
            backtester.print_regime_report(result)
            
            # Show trades if available
            if 'trades_df' in result and len(result['trades_df']) > 0:
                print("\n📝 Sample Trades (first 5):")
                print(result['trades_df'][['entry_date', 'return_pct', 'regime']].head())
            
            print("\n✅ Single backtest with regime analysis working!")
        else:
            print("⚠️  Backtest returned no results (may need data)")
    
    print("\n" + "="*70)
    print("BACKTESTER TEST COMPLETE")
    print("="*70)