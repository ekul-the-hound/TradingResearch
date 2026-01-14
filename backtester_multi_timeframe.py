# backtester_multi_timeframe.py
# Multi-asset, multi-timeframe backtesting engine

import backtrader as bt
import pandas as pd
import time
from datetime import datetime
import config
from database import ResultsDatabase
from data_manager import DataManager

class MultiTimeframeBacktester:
    """
    Tests strategies across multiple assets and timeframes
    """
    
    def __init__(self):
        self.db = ResultsDatabase()
        self.data_manager = DataManager()
        self.results = []
    
    def run_single_backtest(self, strategy_class, symbol, timeframe, 
                           initial_cash=config.DEFAULT_INITIAL_CASH,
                           commission=config.DEFAULT_COMMISSION,
                           strategy_params=None):
        """
        Run a single backtest on one asset/timeframe combination
        """
        
        # Get data from cache
        try:
            data = self.data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                max_bars=config.CANDLE_LIMITS.get(timeframe, 100)
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
        returns = strat.analyzers.returns.get_analysis()
        
        # Calculate metrics
        total_return = ((ending_value - starting_value) / starting_value) * 100
        
        # FIXED: Safe extraction of trade stats using try/except instead of hasattr
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
        
        # Print summary
        sharpe_str = f"{result['sharpe_ratio']:.2f}" if result['sharpe_ratio'] else "N/A"
        print(f"   ✓ {symbol:12} {timeframe:6} | Return: {total_return:+6.2f}% | Sharpe: {sharpe_str:>6} | Trades: {total_trades:3} | Bars: {len(data):4}")
        
        return result
    
    def run_multi_asset_multi_timeframe(self, strategy_class, assets, timeframes,
                                        initial_cash=config.DEFAULT_INITIAL_CASH,
                                        commission=config.DEFAULT_COMMISSION,
                                        strategy_params=None,
                                        save_to_db=True):
        """
        Run backtests across multiple assets and timeframes
        """
        
        total_tests = len(assets) * len(timeframes)
        current_test = 0
        results = []
        
        print(f"\n{'='*80}")
        print(f"MULTI-ASSET MULTI-TIMEFRAME BACKTEST")
        print(f"{'='*80}")
        print(f"Strategy:     {strategy_class.__name__}")
        print(f"Assets:       {len(assets)}")
        print(f"Timeframes:   {len(timeframes)}")
        print(f"Total Tests:  {total_tests}")
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
                    
                    result = self.run_single_backtest(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        timeframe=timeframe,
                        initial_cash=initial_cash,
                        commission=commission,
                        strategy_params=strategy_params
                    )
                    
                    if result:
                        results.append(result)
                        
                        # Save to database if requested
                        if save_to_db:
                            self.db.save_backtest(result)
        
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