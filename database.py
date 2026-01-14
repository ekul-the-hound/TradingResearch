# database.py
# Handles storing and retrieving backtest results

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
        
        # Create backtests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
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
                timestamp TEXT NOT NULL,
                claude_analysis TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized at {self.db_path}")
    
    def save_backtest(self, result):
        """Save a backtest result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtests (
                strategy_name, symbol, timeframe, start_date, end_date, 
                bars_tested, initial_cash, final_value, total_return_pct,
                sharpe_ratio, max_drawdown_pct, total_trades,
                win_rate, profit_factor, strategy_params,
                timestamp, claude_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('strategy_name'),
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
            datetime.now().isoformat(),
            result.get('claude_analysis', '')
        ))
        
        backtest_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return backtest_id
    
    def get_all_backtests(self, strategy_name=None, symbol=None):
        """Retrieve backtests with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM backtests WHERE 1=1"
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_backtest_summary(self):
        """Get summary statistics across all backtests"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_backtests,
                COUNT(DISTINCT strategy_name) as unique_strategies,
                COUNT(DISTINCT symbol) as unique_symbols,
                AVG(total_return_pct) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown_pct) as avg_drawdown
            FROM backtests
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_backtests': result[0],
                'unique_strategies': result[1],
                'unique_symbols': result[2],
                'avg_return': result[3],
                'avg_sharpe': result[4],
                'avg_drawdown': result[5]
            }
        return None
    
    def get_best_performers(self, metric='total_return_pct', limit=10):
        """Get the best performing backtests by a specific metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f'''
            SELECT strategy_name, symbol, {metric}, timestamp
            FROM backtests
            ORDER BY {metric} DESC
            LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        conn.close()
        
        return results