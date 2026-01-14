# backtester.py (Enhanced Version with Claude AI)
# Now with multi-asset testing and AI analysis

import backtrader as bt
import yfinance as yf
import pandas as pd
import time
from anthropic import Anthropic
import json
import config
from database import ResultsDatabase

class StrategyBacktester:
    """
    Enhanced backtester with multi-asset support and AI analysis
    """
    
    def __init__(self):
        self.results = []
        self.db = ResultsDatabase()
        
        # Initialize Claude API
        if config.CLAUDE_API_KEY:
            self.claude = Anthropic(api_key=config.CLAUDE_API_KEY)
            print("✓ Claude AI initialized")
        else:
            self.claude = None
            print("⚠️  Claude AI not initialized (no API key found)")
    
    def download_data_with_retry(self, symbol, start_date, end_date, retries=3):
        """Download data with retry logic"""
        for attempt in range(retries):
            try:
                print(f"📊 Downloading data for {symbol}... (attempt {attempt + 1}/{retries})")
                
                # Create ticker object first
                ticker = yf.Ticker(symbol)
                
                # Download data
                data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
                
                if not data.empty:
                    return data
                else:
                    print(f"⚠️  No data returned, retrying...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"⚠️  Error: {e}")
                if attempt < retries - 1:
                    print(f"   Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"❌ Failed after {retries} attempts")
                    return None
        
        return None
    
    def run_backtest(self, strategy_class, symbol, start_date, end_date, 
                     initial_cash=config.DEFAULT_INITIAL_CASH,
                     commission=config.DEFAULT_COMMISSION,
                     strategy_params=None):
        """Run a backtest on one asset"""
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {symbol}")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Starting Capital: ${initial_cash:,.2f}")
        if strategy_params:
            print(f"Parameters: {strategy_params}")
        print(f"{'='*70}\n")
        
        # Download data
        data = self.download_data_with_retry(symbol, start_date, end_date)
        
        if data is None or data.empty:
            print(f"❌ Could not download data for {symbol}")
            return None
        
        if len(data) < 50:
            print(f"   ⚠️  Too few bars ({len(data)}), minimum 50 required")
            return None
        
        # Rename columns to match backtrader expectations
        data.columns = [col.lower() for col in data.columns]
        
        print(f"✓ Downloaded {len(data)} days of data")
        
        # Create backtrader engine
        cerebro = bt.Cerebro()
        
        # Add data
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
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
        print(f"🔄 Running backtest...")
        starting_value = cerebro.broker.getvalue()
        
        try:
            results = cerebro.run()
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
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
        total_trades = trades.total.closed if trades.total.closed else 0
        
        win_rate = None
        if total_trades > 0:
            wins = trades.won.total if hasattr(trades, 'won') else 0
            win_rate = (wins / total_trades) * 100
        
        profit_factor = None
        if hasattr(trades, 'won') and hasattr(trades, 'lost'):
            gross_profit = trades.won.pnl.total if trades.won.pnl.total else 0
            gross_loss = abs(trades.lost.pnl.total) if trades.lost.pnl.total else 0
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
        
        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {symbol}")
        print(f"{'='*70}")
        print(f"Starting Portfolio:    ${starting_value:,.2f}")
        print(f"Ending Portfolio:      ${ending_value:,.2f}")
        print(f"Total Return:          {total_return:+.2f}%")
        print(f"Sharpe Ratio:          {sharpe.get('sharperatio', 'N/A')}")
        print(f"Max Drawdown:          {drawdown.max.drawdown:.2f}%")
        print(f"Total Trades:          {total_trades}")
        
        if win_rate is not None:
            print(f"Win Rate:              {win_rate:.2f}%")
        if profit_factor is not None:
            print(f"Profit Factor:         {profit_factor:.2f}")
        
        print(f"{'='*70}\n")
        
        # Build result dictionary
        result = {
            'strategy_name': strategy_class.__name__,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'starting_value': starting_value,
            'ending_value': ending_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe.get('sharperatio'),
            'max_drawdown_pct': drawdown.max.drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'strategy_params': strategy_params or {}
        }
        
        return result
    
    def run_multi_asset_backtest(self, strategy_class, symbols, start_date, end_date,
                                  initial_cash=config.DEFAULT_INITIAL_CASH,
                                  commission=config.DEFAULT_COMMISSION,
                                  strategy_params=None):
        """Run backtest across multiple assets"""
        
        print(f"\n{'='*70}")
        print(f"MULTI-ASSET BACKTEST")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Assets: {len(symbols)} symbols")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*70}")
        
        results = []
        
        for symbol in symbols:
            result = self.run_backtest(
                strategy_class=strategy_class,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                commission=commission,
                strategy_params=strategy_params
            )
            
            if result:
                results.append(result)
                # Save to database
                self.db.save_backtest(result)
        
        # Print summary
        self.print_summary(results)
        
        # Get Claude's analysis
        if results and self.claude:
            print("\n🤖 Asking Claude to analyze results...")
            analysis = self.get_claude_analysis(results, strategy_class.__name__)
            print(f"\n{'='*70}")
            print("CLAUDE'S ANALYSIS:")
            print(f"{'='*70}")
            print(analysis)
            print(f"{'='*70}\n")
        elif not self.claude:
            print("\n⚠️  Skipping Claude analysis (API key not configured)")
        
        return results
    
    def print_summary(self, results):
        """Print summary statistics"""
        if not results:
            print("\n❌ No results to summarize")
            return
        
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        
        # Calculate averages
        avg_return = sum(r['total_return_pct'] for r in results) / len(results)
        
        sharpe_results = [r['sharpe_ratio'] for r in results if r['sharpe_ratio'] is not None]
        avg_sharpe = sum(sharpe_results) / len(sharpe_results) if sharpe_results else 0
        
        avg_dd = sum(r['max_drawdown_pct'] for r in results) / len(results)
        avg_trades = sum(r['total_trades'] for r in results) / len(results)
        
        positive_returns = len([r for r in results if r['total_return_pct'] > 0])
        
        print(f"Assets Tested:        {len(results)}")
        print(f"Positive Returns:     {positive_returns}/{len(results)} ({positive_returns/len(results)*100:.1f}%)")
        print(f"Average Return:       {avg_return:.2f}%")
        print(f"Average Sharpe:       {avg_sharpe:.2f}")
        print(f"Average Max DD:       {avg_dd:.2f}%")
        print(f"Average Trades:       {avg_trades:.1f}")
        
        # Best and worst performers
        best = max(results, key=lambda x: x['total_return_pct'])
        worst = min(results, key=lambda x: x['total_return_pct'])
        
        print(f"\nBest Performer:       {best['symbol']} ({best['total_return_pct']:.2f}%)")
        print(f"Worst Performer:      {worst['symbol']} ({worst['total_return_pct']:.2f}%)")
        print(f"{'='*70}")
    
    def get_claude_analysis(self, results, strategy_name):
        """Use Claude to analyze backtest results"""
        
        # Prepare results summary for Claude
        summary = {
            'strategy_name': strategy_name,
            'total_assets_tested': len(results),
            'results': []
        }
        
        for r in results:
            summary['results'].append({
                'symbol': r['symbol'],
                'return_pct': round(r['total_return_pct'], 2),
                'sharpe': round(r['sharpe_ratio'], 2) if r['sharpe_ratio'] else None,
                'max_drawdown_pct': round(r['max_drawdown_pct'], 2),
                'trades': r['total_trades'],
                'win_rate': round(r['win_rate'], 2) if r['win_rate'] else None,
                'profit_factor': round(r['profit_factor'], 2) if r['profit_factor'] else None
            })
        
        # Create prompt for Claude
        prompt = f"""I just backtested a trading strategy called "{strategy_name}" across {len(results)} different assets.

Here are the results:

{json.dumps(summary, indent=2)}

Please analyze these results and provide:

1. Overall assessment: Is this strategy promising or not?
2. Performance patterns: Which types of assets performed best/worst?
3. Key concerns: What are the biggest red flags in these results?
4. Specific insights: What do the metrics tell us about the strategy's behavior?
5. Next steps: What should I test or modify next?

Be direct and honest. If the strategy looks bad, say so clearly."""

        # Call Claude API
        try:
            message = self.claude.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            analysis = message.content[0].text
            return analysis
            
        except Exception as e:
            return f"Error getting Claude analysis: {str(e)}"