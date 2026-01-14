# ==============================================================================
# config.py - Trading Data Pipeline Configuration
# ==============================================================================
# COMPLETE VERSION with all required variables
# 
# CHANGELOG:
# - ADDED: FOREX_ENABLED, CRYPTO_ENABLED, INDICES_ENABLED, COMMODITIES_ENABLED flags
# - ADDED: DATA_CACHE_PATH and CACHE_SUBDIRS for data_manager.py
# - ADDED: DATABASE_PATH for database operations
# - ADDED: DEFAULT_INITIAL_CASH and DEFAULT_COMMISSION for backtesting
# - RESTORED: INDEX_WATCHLIST and COMMODITY_WATCHLIST (commented but defined)
# - FIXED: All references now properly defined
# ==============================================================================

import os
from pathlib import Path

# ==============================================================================
# CLAUDE API CONFIGURATION
# ==============================================================================
def load_api_key():
    """Load API key from file, fallback to environment variable"""
    key_file = Path(__file__).parent / 'BacktestingAgent_API_KEY.txt'
    if key_file.exists():
        with open(key_file, 'r') as f:
            key = f.read().strip()
            if key:
                return key
    return os.getenv('ANTHROPIC_API_KEY', '')

CLAUDE_API_KEY = load_api_key()
CLAUDE_MODEL = 'claude-sonnet-4-20250514'
CLAUDE_MAX_TOKENS = 4096

# ==============================================================================
# BASE DIRECTORY CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).parent

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
DB_PATH = BASE_DIR / 'data' / 'trading_data.db'
ANALYSIS_DB_PATH = BASE_DIR / 'data' / 'analysis_results.db'
DATABASE_PATH = BASE_DIR / 'results' / 'backtest_results.db'  # Main results database

# Create directories
(BASE_DIR / 'data').mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'results').mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'logs').mkdir(parents=True, exist_ok=True)

# ==============================================================================
# BACKTESTING DEFAULTS
# ==============================================================================
DEFAULT_INITIAL_CASH = 10000
DEFAULT_COMMISSION = 0.001  # 0.1%

# ==============================================================================
# ASSET CLASS ENABLE/DISABLE FLAGS
# ==============================================================================
FOREX_ENABLED = True        # Local files from E:/TradingData/forex
CRYPTO_ENABLED = True       # CCXT (Binance -> Hyperliquid)
INDICES_ENABLED = False     # Disabled - awaiting IBKR integration
COMMODITIES_ENABLED = False # Disabled - awaiting futures data source

# ==============================================================================
# FOREX WATCHLIST - ACTIVE
# ==============================================================================
FOREX_WATCHLIST = [
    'EUR-USD',
    'GBP-USD', 
    'USD-JPY',
    'AUD-USD',
    'USD-CAD',
    'USD-CHF',
    'NZD-USD'
]

# Forex ticker mapping for HistData.com file naming convention
# Maps watchlist symbols to file name formats (e.g., EUR-USD -> EURUSD)
FOREX_TICKERS = {
    'EUR-USD': 'EURUSD',
    'GBP-USD': 'GBPUSD',
    'USD-JPY': 'USDJPY',
    'AUD-USD': 'AUDUSD',
    'USD-CAD': 'USDCAD',
    'USD-CHF': 'USDCHF',
    'NZD-USD': 'NZDUSD'
}

# File naming pattern for local Forex files
# Example: DAT_XLSX_EURUSD_M1_2020.xlsx
FOREX_FILE_PATTERN = "DAT_XLSX_{ticker}_M1_{year}.xlsx"

# Expected column names in HistData.com files
FOREX_COLUMNS = {
    'datetime': 'datetime',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume'
}

# ==============================================================================
# CRYPTO WATCHLIST - ACTIVE
# ==============================================================================
CRYPTO_WATCHLIST = [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'XRP-USD',
    'ADA-USD'
]

# Crypto symbol mapping for different exchanges
# Translates friendly names to exchange-specific formats
CRYPTO_SYMBOL_MAP = {
    'BTC-USD': {'binance': 'BTC/USDT', 'hyperliquid': 'BTC'},
    'ETH-USD': {'binance': 'ETH/USDT', 'hyperliquid': 'ETH'},
    'SOL-USD': {'binance': 'SOL/USDT', 'hyperliquid': 'SOL'},
    'XRP-USD': {'binance': 'XRP/USDT', 'hyperliquid': 'XRP'},
    'ADA-USD': {'binance': 'ADA/USDT', 'hyperliquid': 'ADA'}
}

# ==============================================================================
# INDICES WATCHLIST - DISABLED (preserved for future IBKR integration)
# ==============================================================================
INDEX_WATCHLIST = [
    # '^GSPC',   # S&P 500
    # '^DJI',    # Dow Jones Industrial Average
    # '^IXIC',   # NASDAQ Composite
    # '^RUT',    # Russell 2000
    # '^VIX',    # CBOE Volatility Index
    # '^FTSE',   # FTSE 100 (UK)
    # '^GDAXI',  # DAX (Germany)
    # '^N225',   # Nikkei 225 (Japan)
]

# ==============================================================================
# COMMODITIES WATCHLIST - DISABLED (preserved for future use)
# ==============================================================================
COMMODITY_WATCHLIST = [
    # 'GC=F',    # Gold Futures
    # 'SI=F',    # Silver Futures
    # 'CL=F',    # Crude Oil WTI Futures
    # 'BZ=F',    # Brent Crude Oil Futures
    # 'NG=F',    # Natural Gas Futures
    # 'HG=F',    # Copper Futures
]

# ==============================================================================
# STOCK WATCHLIST - For legacy backtester.py (Phase 1/2 tests)
# ==============================================================================
STOCK_WATCHLIST = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Google
    'AMZN',   # Amazon
    'TSLA',   # Tesla
    'NVDA',   # NVIDIA
    'META',   # Meta
    'JPM',    # JP Morgan
    'V',      # Visa
    'WMT'     # Walmart
]

# ==============================================================================
# ALL ASSETS CONFIGURATION
# ==============================================================================
ALL_ASSETS = {
    'forex': FOREX_WATCHLIST if FOREX_ENABLED else [],
    'crypto': CRYPTO_WATCHLIST if CRYPTO_ENABLED else [],
    'indices': INDEX_WATCHLIST if INDICES_ENABLED else [],
    'commodities': COMMODITY_WATCHLIST if COMMODITIES_ENABLED else [],
}

# ==============================================================================
# TIMEFRAME DEFINITIONS
# ==============================================================================
TIMEFRAMES = {
    '1min': '1m',
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
    '1hour': '1h',
    '4hour': '4h',
    '1day': '1d',
    '1week': '1w',
    '1month': '1M'
}

# ==============================================================================
# ALLOWED TIMEFRAMES PER ASSET CLASS
# ==============================================================================
FOREX_ALLOWED_TIMEFRAMES = ['1min', '5min', '15min', '30min', '1hour', '4hour', '1day']
CRYPTO_ALLOWED_TIMEFRAMES = ['5min', '15min', '30min', '1hour', '4hour', '1day']
INDEX_ALLOWED_TIMEFRAMES = ['1hour', '4hour', '1day', '1week', '1month']
COMMODITY_ALLOWED_TIMEFRAMES = ['1hour', '4hour', '1day', '1week', '1month']

# ==============================================================================
# DATA SOURCE PRIORITIES
# ==============================================================================
FOREX_SOURCES = ['local']       # Local files only - no yfinance fallback
CRYPTO_SOURCES = ['ccxt']       # CCXT only - no yfinance fallback
INDEX_SOURCES = ['yfinance']    # Would use yfinance if re-enabled
COMMODITY_SOURCES = ['yfinance'] # Would use yfinance if re-enabled

# ==============================================================================
# CACHE CONFIGURATION
# ==============================================================================
DATA_CACHE_PATH = Path('E:/TradingData')
CACHE_DIR = DATA_CACHE_PATH / 'cache'

# Cache subdirectories by asset type
CACHE_SUBDIRS = {
    'forex': str(DATA_CACHE_PATH / 'forex'),
    'crypto': str(CACHE_DIR / 'crypto'),
    'indices': str(CACHE_DIR / 'indices'),
    'commodities': str(CACHE_DIR / 'commodities'),
    'raw': str(CACHE_DIR / 'raw')
}

# Create all cache directories
for cache_path in CACHE_SUBDIRS.values():
    Path(cache_path).mkdir(parents=True, exist_ok=True)

# Convenience aliases
FOREX_CACHE_DIR = CACHE_SUBDIRS['forex']
CRYPTO_CACHE_DIR = CACHE_SUBDIRS['crypto']

# ==============================================================================
# FOREX CONFIGURATION - LOCAL FILES ONLY
# ==============================================================================
FOREX_BASE_PATH = "E:/TradingData/forex"
FOREX_BASE_TIMEFRAME = '1min'  # Base timeframe in local files

# ==============================================================================
# CRYPTO CONFIGURATION - CCXT ONLY
# ==============================================================================
CRYPTO_EXCHANGE_PRIORITY = ['binance', 'hyperliquid']

# Exchange-specific toggles
BINANCE_ENABLED = True
HYPERLIQUID_ENABLED = True

# API credentials (optional for public data)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
HYPERLIQUID_API_KEY = os.getenv('HYPERLIQUID_API_KEY', '')
HYPERLIQUID_API_SECRET = os.getenv('HYPERLIQUID_API_SECRET', '')

# ==============================================================================
# DATA LIMITS PER SOURCE (REALISTIC PRODUCTION VALUES)
# ==============================================================================
CANDLE_LIMITS = {
    '1min': 500000,   # ~347 days of 1-min data (24/7 market)
    '5min': 200000,   # ~694 days of 5-min data
    '15min': 100000,  # ~1041 days of 15-min data
    '30min': 100000,  # ~2083 days of 30-min data
    '1hour': 50000,   # ~2083 days of 1-hour data
    '4hour': 25000,   # ~4166 days of 4-hour data
    '1day': 10000,    # ~27 years of daily data
    '1week': 5000,    # ~96 years of weekly data
    '1month': 2000    # ~166 years of monthly data
}

# ==============================================================================
# TECHNICAL INDICATORS (for future feature engineering)
# ==============================================================================
TECHNICAL_INDICATORS = [
    'sma_20', 'sma_50', 'sma_200',
    'ema_12', 'ema_26',
    'rsi_14',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14',
    'obv'
]

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / 'logs' / 'trading_pipeline.log'

# ==============================================================================
# VALIDATION
# ==============================================================================
def validate_config():
    """
    Validates configuration settings
    """
    errors = []
    warnings = []
    
    # Check required directories exist
    if FOREX_ENABLED and not Path(FOREX_BASE_PATH).exists():
        warnings.append(f"Forex data directory not found: {FOREX_BASE_PATH}")
    
    # Verify cache directories were created
    if not Path(DATA_CACHE_PATH).exists():
        errors.append(f"Data cache path does not exist: {DATA_CACHE_PATH}")
    
    # Check for Claude API key
    if not CLAUDE_API_KEY:
        warnings.append("Claude API key not found (check BacktestingAgent_API_KEY.txt)")
    
    # Check CCXT availability if crypto enabled
    if CRYPTO_ENABLED:
        try:
            import ccxt
        except ImportError:
            warnings.append("CCXT not installed. Run: pip install ccxt")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Configuration validated successfully")
    return True

def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    
    print(f"\nActive Asset Classes:")
    print(f"  {'✅' if FOREX_ENABLED else '⏸️ '} Forex:       {len(FOREX_WATCHLIST)} pairs (local files)")
    print(f"  {'✅' if CRYPTO_ENABLED else '⏸️ '} Crypto:      {len(CRYPTO_WATCHLIST)} currencies (CCXT)")
    print(f"  {'✅' if INDICES_ENABLED else '⏸️ '} Indices:     {len(INDEX_WATCHLIST)} indices (disabled)")
    print(f"  {'✅' if COMMODITIES_ENABLED else '⏸️ '} Commodities: {len(COMMODITY_WATCHLIST)} commodities (disabled)")
    
    print(f"\nData Paths:")
    print(f"  Forex data:  {FOREX_BASE_PATH}")
    print(f"  Cache:       {DATA_CACHE_PATH}")
    print(f"  Database:    {DATABASE_PATH}")
    
    print(f"\nAPI Status:")
    print(f"  Claude API:  {'✅ Configured' if CLAUDE_API_KEY else '❌ Not configured'}")
    
    print("="*70)

if __name__ == "__main__":
    print_config_summary()
    print()
    validate_config()