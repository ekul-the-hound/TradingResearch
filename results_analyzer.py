# results_analyzer.py
# Query and analyze backtest results from database

import sqlite3
import pandas as pd
import config

class ResultsAnalyzer:
    """
    Tool for querying and analyzing backtest results
    """
    
    def __init__(self):
        self.db_path = config.DATABASE_PATH
    
    def get_all_results(self):
        """Get all backtest results as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM backtests", conn)
        conn.close()
        return df
    
    def filter_by_sharpe(self, min_sharpe=0.5):
        """
        Get results with Sharpe ratio above threshold
        Default: 0.5 (your specified threshold)
        """
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT * FROM backtests
            WHERE sharpe_ratio > {min_sharpe}
            ORDER BY sharpe_ratio DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def top_performers(self, n=10, metric='total_return_pct'):
        """Get top N performers by metric"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT * FROM backtests
            ORDER BY {metric} DESC
            LIMIT {n}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def worst_performers(self, n=10, metric='total_return_pct'):
        """Get worst N performers by metric"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT * FROM backtests
            ORDER BY {metric} ASC
            LIMIT {n}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def by_timeframe(self, timeframe):
        """Get all results for a specific timeframe"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM backtests
            WHERE timeframe = ?
            ORDER BY total_return_pct DESC
        """
        df = pd.read_sql_query(query, conn, params=(timeframe,))
        conn.close()
        return df
    
    def by_asset(self, symbol):
        """Get all results for a specific asset"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM backtests
            WHERE symbol = ?
            ORDER BY total_return_pct DESC
        """
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        return df
    
    def by_asset_class(self, asset_class):
        """
        Get results for an asset class
        asset_class: 'Forex', 'Indices', 'Commodities', or 'Crypto'
        """
        if asset_class == 'Forex':
            assets = config.FOREX_WATCHLIST
        elif asset_class == 'Indices':
            assets = config.INDEX_WATCHLIST
        elif asset_class == 'Commodities':
            assets = config.COMMODITY_WATCHLIST
        elif asset_class == 'Crypto':
            assets = config.CRYPTO_WATCHLIST
        else:
            return pd.DataFrame()
        
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join('?' * len(assets))
        query = f"""
            SELECT * FROM backtests
            WHERE symbol IN ({placeholders})
            ORDER BY total_return_pct DESC
        """
        df = pd.read_sql_query(query, conn, params=assets)
        conn.close()
        return df
    
    def summary_by_timeframe(self):
        """Get summary statistics grouped by timeframe"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT 
                timeframe,
                COUNT(*) as total_tests,
                AVG(total_return_pct) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown_pct) as avg_drawdown,
                AVG(total_trades) as avg_trades,
                SUM(CASE WHEN total_return_pct > 0 THEN 1 ELSE 0 END) as positive_count
            FROM backtests
            GROUP BY timeframe
            ORDER BY avg_return DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def summary_by_asset_class(self):
        """Get summary statistics grouped by asset class"""
        df_all = self.get_all_results()
        
        results = []
        for asset_class in ['Forex', 'Indices', 'Commodities', 'Crypto']:
            df_class = self.by_asset_class(asset_class)
            if not df_class.empty:
                results.append({
                    'asset_class': asset_class,
                    'total_tests': len(df_class),
                    'avg_return': df_class['total_return_pct'].mean(),
                    'avg_sharpe': df_class['sharpe_ratio'].mean(),
                    'avg_drawdown': df_class['max_drawdown_pct'].mean(),
                    'positive_count': (df_class['total_return_pct'] > 0).sum()
                })
        
        return pd.DataFrame(results)
    
    def strong_performers_report(self, min_sharpe=0.5, min_trades=10):
        """
        Generate report of strong performers
        Default: Sharpe > 0.5, at least 10 trades
        """
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT * FROM backtests
            WHERE sharpe_ratio > {min_sharpe}
            AND total_trades >= {min_trades}
            ORDER BY sharpe_ratio DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

def print_dataframe(df, title=None, max_rows=20):
    """Pretty print a DataFrame"""
    if title:
        print(f"\n{'='*80}")
        print(title)
        print(f"{'='*80}")
    
    if df.empty:
        print("  (No results)")
        return
    
    # Select key columns for display
    display_cols = ['symbol', 'timeframe', 'total_return_pct', 'sharpe_ratio', 
                   'max_drawdown_pct', 'total_trades', 'win_rate']
    display_cols = [col for col in display_cols if col in df.columns]
    
    # Format numeric columns
    pd.options.display.float_format = '{:.2f}'.format
    
    if len(df) > max_rows:
        print(f"\n  Showing top {max_rows} of {len(df)} results:\n")
        print(df[display_cols].head(max_rows).to_string(index=False))
    else:
        print(f"\n  {len(df)} results:\n")
        print(df[display_cols].to_string(index=False))

def main():
    """
    Interactive results analyzer
    """
    
    analyzer = ResultsAnalyzer()
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS ANALYZER")
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        print("QUERY OPTIONS:")
        print("-"*80)
        print("  1. Show strong performers (Sharpe > 0.5)")
        print("  2. Top 10 performers by return")
        print("  3. Worst 10 performers")
        print("  4. Results by timeframe")
        print("  5. Results by asset")
        print("  6. Results by asset class")
        print("  7. Summary by timeframe")
        print("  8. Summary by asset class")
        print("  9. Custom Sharpe filter")
        print("  0. Exit")
        print("-"*80)
        
        choice = input("\nSelect option (0-9): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            df = analyzer.strong_performers_report(min_sharpe=0.5, min_trades=10)
            print_dataframe(df, "Strong Performers (Sharpe > 0.5, Trades >= 10)")
            
            if not df.empty:
                print(f"\n  Found {len(df)} strong performers!")
                print(f"  Average Sharpe: {df['sharpe_ratio'].mean():.2f}")
                print(f"  Average Return: {df['total_return_pct'].mean():+.2f}%")
        
        elif choice == '2':
            df = analyzer.top_performers(n=10)
            print_dataframe(df, "Top 10 Performers by Return")
        
        elif choice == '3':
            df = analyzer.worst_performers(n=10)
            print_dataframe(df, "Worst 10 Performers by Return")
        
        elif choice == '4':
            print("\nAvailable timeframes: 1min, 5min, 15min, 30min, 1hour, 4hour")
            timeframe = input("Enter timeframe: ").strip()
            df = analyzer.by_timeframe(timeframe)
            print_dataframe(df, f"Results for {timeframe}")
            
            if not df.empty:
                print(f"\n  Average Return: {df['total_return_pct'].mean():+.2f}%")
                print(f"  Positive: {(df['total_return_pct'] > 0).sum()}/{len(df)}")
        
        elif choice == '5':
            print("\nEnter asset symbol (e.g., BTC-USD, EURUSD=X, ^GSPC, GC=F)")
            symbol = input("Symbol: ").strip()
            df = analyzer.by_asset(symbol)
            print_dataframe(df, f"Results for {symbol}")
            
            if not df.empty:
                print(f"\n  Average Return: {df['total_return_pct'].mean():+.2f}%")
                print(f"  Best Timeframe: {df.iloc[0]['timeframe']} ({df.iloc[0]['total_return_pct']:+.2f}%)")
        
        elif choice == '6':
            print("\nAsset classes: Forex, Indices, Commodities, Crypto")
            asset_class = input("Enter asset class: ").strip().title()
            df = analyzer.by_asset_class(asset_class)
            print_dataframe(df, f"{asset_class} Results")
            
            if not df.empty:
                print(f"\n  Average Return: {df['total_return_pct'].mean():+.2f}%")
                print(f"  Positive: {(df['total_return_pct'] > 0).sum()}/{len(df)}")
        
        elif choice == '7':
            df = analyzer.summary_by_timeframe()
            print(f"\n{'='*80}")
            print("SUMMARY BY TIMEFRAME")
            print(f"{'='*80}\n")
            print(df.to_string(index=False))
        
        elif choice == '8':
            df = analyzer.summary_by_asset_class()
            print(f"\n{'='*80}")
            print("SUMMARY BY ASSET CLASS")
            print(f"{'='*80}\n")
            print(df.to_string(index=False))
        
        elif choice == '9':
            try:
                min_sharpe = float(input("Enter minimum Sharpe ratio (default 0.5): ").strip() or 0.5)
                df = analyzer.filter_by_sharpe(min_sharpe=min_sharpe)
                print_dataframe(df, f"Results with Sharpe > {min_sharpe}")
                
                if not df.empty:
                    print(f"\n  Found {len(df)} results above threshold")
            except ValueError:
                print("  Invalid input. Please enter a number.")
        
        else:
            print("  Invalid option. Please try again.")

if __name__ == "__main__":
    main()