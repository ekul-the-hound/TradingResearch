# ==============================================================================
# database.py
# ==============================================================================
# Handles storing and retrieving backtest results
# UPDATED: Added variant_id support for mutation agent
# ==============================================================================

import sqlite3
import json
from datetime import datetime
import config
import os

class ResultsDatabase:
    """
    Stores all backtest results in a SQLite database
    This keeps a permanent record of every test you run
    """
    
    def __init__(self, db_path=config.DATABASE_PATH):
        self.db_path = db_path
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create backtest_results table (new name for consistency)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                variant_id TEXT,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                bars_tested INTEGER,
                initial_cash REAL,
                final_value REAL,
                total_return_pct REAL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                strategy_params TEXT,
                modifications TEXT,
                timestamp TEXT NOT NULL,
                claude_analysis TEXT
            )
        ''')
        
        # Also create the old table name for backwards compatibility
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                variant_id TEXT,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                bars_tested INTEGER,
                initial_cash REAL,
                final_value REAL,
                total_return_pct REAL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                strategy_params TEXT,
                modifications TEXT,
                timestamp TEXT NOT NULL,
                claude_analysis TEXT
            )
        ''')
        
        # Create index for faster variant queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_variant_id 
            ON backtest_results(variant_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_strategy_name 
            ON backtest_results(strategy_name)
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized at {self.db_path}")
    
    def save_backtest(self, result):
        """Save a backtest result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save to backtest_results table
        cursor.execute('''
            INSERT INTO backtest_results (
                strategy_name, variant_id, symbol, timeframe, start_date, end_date, 
                bars_tested, initial_cash, final_value, total_return_pct,
                sharpe_ratio, max_drawdown_pct, total_trades,
                win_rate, profit_factor, strategy_params, modifications,
                timestamp, claude_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('strategy_name'),
            result.get('variant_id'),
            result.get('symbol'),
            result.get('timeframe'),
            result.get('start_date'),
            result.get('end_date'),
            result.get('bars_tested'),
            result.get('starting_value'),
            result.get('ending_value'),
            result.get('total_return_pct'),
            result.get('sharpe_ratio'),
            result.get('max_drawdown_pct'),
            result.get('total_trades'),
            result.get('win_rate'),
            result.get('profit_factor'),
            json.dumps(result.get('strategy_params', {})),
            json.dumps(result.get('modifications', [])),
            datetime.now().isoformat(),
            result.get('claude_analysis', '')
        ))
        
        backtest_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return backtest_id
    
    def get_all_backtests(self, strategy_name=None, symbol=None, variant_id=None):
        """Retrieve backtests with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM backtest_results WHERE 1=1"
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if variant_id:
            query += " AND variant_id = ?"
            params.append(variant_id)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_variant_results(self, variant_id):
        """Get all results for a specific variant"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM backtest_results 
            WHERE variant_id = ?
            ORDER BY total_return_pct DESC
        ''', (variant_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_all_variants(self):
        """Get list of all unique variants"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT variant_id, strategy_name 
            FROM backtest_results 
            WHERE variant_id IS NOT NULL
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_variant_summary(self, variant_id):
        """Get summary statistics for a specific variant"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_tests,
                AVG(total_return_pct) as avg_return,
                MAX(total_return_pct) as best_return,
                MIN(total_return_pct) as worst_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(win_rate) as avg_win_rate,
                AVG(max_drawdown_pct) as avg_drawdown
            FROM backtest_results
            WHERE variant_id = ?
        ''', (variant_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_tests': result[0],
                'avg_return': result[1],
                'best_return': result[2],
                'worst_return': result[3],
                'avg_sharpe': result[4],
                'avg_win_rate': result[5],
                'avg_drawdown': result[6]
            }
        return None
    
    def get_backtest_summary(self):
        """Get summary statistics across all backtests"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_backtests,
                COUNT(DISTINCT strategy_name) as unique_strategies,
                COUNT(DISTINCT variant_id) as unique_variants,
                COUNT(DISTINCT symbol) as unique_symbols,
                AVG(total_return_pct) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown_pct) as avg_drawdown
            FROM backtest_results
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_backtests': result[0],
                'unique_strategies': result[1],
                'unique_variants': result[2],
                'unique_symbols': result[3],
                'avg_return': result[4],
                'avg_sharpe': result[5],
                'avg_drawdown': result[6]
            }
        return None
    
    def get_best_performers(self, metric='total_return_pct', limit=10):
        """Get the best performing backtests by a specific metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f'''
            SELECT strategy_name, variant_id, symbol, timeframe, {metric}, timestamp
            FROM backtest_results
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def compare_variants(self):
        """Compare all variants by average return"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COALESCE(variant_id, strategy_name) as variant,
                strategy_name,
                COUNT(*) as tests,
                AVG(total_return_pct) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(win_rate) as avg_win_rate,
                SUM(CASE WHEN total_return_pct > 0 THEN 1 ELSE 0 END) as positive_tests
            FROM backtest_results
            GROUP BY COALESCE(variant_id, strategy_name)
            ORDER BY avg_return DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def clear_variant_results(self, variant_id):
        """Clear all results for a specific variant"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM backtest_results WHERE variant_id = ?
        ''', (variant_id,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
    
    def clear_all_variants(self):
        """Clear all variant results (keep base strategy results)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM backtest_results WHERE variant_id IS NOT NULL
        ''')
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
