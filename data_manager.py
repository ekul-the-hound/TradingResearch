# ==============================================================================
# data_manager.py
# ==============================================================================
# Unified data manager for the Trading Research system
#
# Data Sources:
# - Forex: Local CSV files (E:/TradingData/forex) - merged 1-min files
# - Crypto: LOCAL FILES FIRST (E:/TradingData/crypto), then CCXT fallback
# - Indices: LOCAL FILES (E:/TradingData/indices) - Kaggle downloads
# - Commodities: DISABLED (awaiting futures data source)
#
# UPDATED: Added support for local Kaggle downloads for crypto and indices
#
# Features:
# - Automatic resampling of 1-minute data to higher timeframes
# - Smart file detection (handles various CSV column formats)
# - CCXT fallback for crypto if local files not found
# - Caching system for performance
# ==============================================================================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
import time
import config

# CCXT for crypto data (fallback)
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️  CCXT not installed. Run: pip install ccxt")


class DataManager:
    """
    Unified data manager for all asset classes
    
    Usage:
        manager = DataManager()
        data = manager.get_data('EUR-USD', timeframe='1hour', max_bars=1000)
        data = manager.get_data('BTC-USD', timeframe='4hour', max_bars=500)  # Local first, CCXT fallback
        data = manager.get_data('SPX', timeframe='1day', max_bars=1000)  # Local Kaggle files
    """
    
    def __init__(self):
        self._ensure_cache_dirs()
        self.exchanges = {}
        self._local_file_cache = {'crypto': [], 'indices': []}
        
        # Scan for local files first
        self._scan_local_files()
        
        # Initialize CCXT exchanges if available and crypto enabled (as fallback)
        if CCXT_AVAILABLE and config.CRYPTO_ENABLED:
            self._init_exchanges()
    
    def _ensure_cache_dirs(self):
        """Create cache directories if they don't exist"""
        for subdir in config.CACHE_SUBDIRS.values():
            Path(subdir).mkdir(parents=True, exist_ok=True)
    
    def _scan_local_files(self):
        """Scan data directories and cache file locations"""
        
        # Scan crypto directory
        crypto_path = Path(config.DATA_CACHE_PATH) / 'crypto'
        if crypto_path.exists():
            self._local_file_cache['crypto'] = self._find_data_files(crypto_path)
            if self._local_file_cache['crypto']:
                print(f"✅ Found {len(self._local_file_cache['crypto'])} local crypto files")
        
        # Scan indices directory
        indices_path = Path(config.DATA_CACHE_PATH) / 'indices'
        if indices_path.exists():
            self._local_file_cache['indices'] = self._find_data_files(indices_path)
            if self._local_file_cache['indices']:
                print(f"✅ Found {len(self._local_file_cache['indices'])} local indices files")
    
    def _find_data_files(self, directory):
        """Find all CSV and parquet files in a directory"""
        files = []
        for ext in ['*.csv', '*.CSV', '*.parquet', '*.xlsx']:
            files.extend(glob.glob(str(directory / '**' / ext), recursive=True))
            files.extend(glob.glob(str(directory / ext)))
        return list(set(files))
    
    def _init_exchanges(self):
        """Initialize CCXT exchanges"""
        self.exchanges = {}
        
        if config.BINANCE_ENABLED and 'binance' in config.CRYPTO_EXCHANGE_PRIORITY:
            try:
                exchange_config = {
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
                if config.BINANCE_API_KEY:
                    exchange_config['apiKey'] = config.BINANCE_API_KEY
                    exchange_config['secret'] = config.BINANCE_API_SECRET
                
                self.exchanges['binance'] = ccxt.binanceus(exchange_config)
                print("✅ Binance US exchange initialized (fallback)")
            except Exception as e:
                print(f"⚠️  Failed to initialize Binance US: {e}")
        
        # Kraken as backup
        try:
            self.exchanges['kraken'] = ccxt.kraken({'enableRateLimit': True})
            print("✅ Kraken exchange initialized (fallback)")
        except Exception as e:
            print(f"⚠️  Failed to initialize Kraken: {e}")
        
        if config.HYPERLIQUID_ENABLED and 'hyperliquid' in config.CRYPTO_EXCHANGE_PRIORITY:
            try:
                self.exchanges['hyperliquid'] = ccxt.hyperliquid({'enableRateLimit': True})
                print("✅ Hyperliquid exchange initialized (fallback)")
            except Exception as e:
                print(f"⚠️  Failed to initialize Hyperliquid: {e}")
    
    def _determine_asset_type(self, symbol):
        """
        Determine the asset class for a given symbol
        
        Returns: 'forex', 'crypto', 'indices', 'commodities', or 'unknown'
        """
        # Check watchlists first
        if symbol in config.FOREX_WATCHLIST:
            return 'forex'
        elif symbol in config.CRYPTO_WATCHLIST:
            return 'crypto'
        elif symbol in config.INDEX_WATCHLIST:
            return 'indices'
        elif symbol in config.COMMODITY_WATCHLIST:
            return 'commodities'
        
        # Fallback pattern detection
        if symbol.endswith('=X') or '-' in symbol and symbol.replace('-', '') in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']:
            return 'forex'
        elif symbol.endswith('-USD') or symbol.endswith('USDT') or symbol in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']:
            return 'crypto'
        elif symbol.startswith('^') or symbol in ['SPX', 'NDX', 'DJI', 'SPY', 'QQQ', 'IWM']:
            return 'indices'
        elif symbol.endswith('=F'):
            return 'commodities'
        
        return 'unknown'
    
    def get_data(self, symbol, timeframe='1hour', max_bars=None, use_cache=True, **kwargs):
        """
        Get OHLCV data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EUR-USD', 'BTC-USD', 'SPX')
            timeframe: '1min', '5min', '15min', '30min', '1hour', '4hour', '1day'
            max_bars: Maximum number of bars to return
            use_cache: Whether to use cached data
            **kwargs: Additional arguments (for backward compatibility with 'periods')
        
        Returns:
            DataFrame with OHLCV data or None if unavailable
        """
        # Handle legacy 'periods' parameter
        if 'periods' in kwargs and max_bars is None:
            max_bars = kwargs['periods']
        
        asset_type = self._determine_asset_type(symbol)
        
        # ====================================================================
        # INDICES: Local files from Kaggle
        # ====================================================================
        if asset_type == 'indices':
            if not config.INDICES_ENABLED:
                print(f"⏸️  Indices disabled in config")
                return None
            return self._get_local_indices_data(symbol, timeframe, max_bars, use_cache)
        
        # ====================================================================
        # COMMODITIES: DISABLED
        # ====================================================================
        if asset_type == 'commodities':
            if not config.COMMODITIES_ENABLED:
                print(f"⏸️  Data not added yet: {symbol} {timeframe}")
                print(f"   (Commodities awaiting futures data source)")
                return None
        
        # ====================================================================
        # FOREX: Local files only
        # ====================================================================
        if asset_type == 'forex':
            if not config.FOREX_ENABLED:
                print(f"⏸️  Forex disabled in config")
                return None
            return self._get_forex_data(symbol, timeframe, max_bars, use_cache)
        
        # ====================================================================
        # CRYPTO: Local files first, CCXT fallback
        # ====================================================================
        if asset_type == 'crypto':
            if not config.CRYPTO_ENABLED:
                print(f"⏸️  Crypto disabled in config")
                return None
            
            # Try local files first
            local_data = self._get_local_crypto_data(symbol, timeframe, max_bars, use_cache)
            if local_data is not None and len(local_data) > 0:
                return local_data
            
            # Fall back to CCXT
            if CCXT_AVAILABLE and self.exchanges:
                print(f"   📡 No local data, trying CCXT...")
                return self._get_crypto_data_ccxt(symbol, timeframe, max_bars, use_cache)
            else:
                print(f"❌ No local crypto data and CCXT unavailable for {symbol}")
                return None
        
        # ====================================================================
        # UNKNOWN ASSET TYPE
        # ====================================================================
        print(f"❌ Unknown asset type for {symbol}")
        return None
    
    # ========================================================================
    # LOCAL FILE LOADING (CRYPTO & INDICES)
    # ========================================================================
    
    def _find_local_file(self, symbol, asset_type):
        """
        Find a local file matching the symbol
        
        Handles various naming conventions:
        - BTC-USD, BTCUSD, BTC_USD, btc-usd
        - SPX, ^GSPC, SP500, spx
        """
        files = self._local_file_cache.get(asset_type, [])
        
        # Clean symbol for matching
        symbol_variants = [
            symbol,
            symbol.replace('-', ''),
            symbol.replace('-', '_'),
            symbol.replace('-USD', 'USDT'),
            symbol.replace('-USD', ''),
            symbol.lower(),
            symbol.upper(),
        ]
        
        # Add index-specific variants
        if asset_type == 'indices':
            index_map = {
                'SPX': ['^GSPC', 'SP500', 'SPX500', 'S&P500'],
                'NDX': ['^IXIC', 'NASDAQ', 'NDX100', 'NASDAQ100'],
                'DJI': ['^DJI', 'DJIA', 'DOW', 'DOW30'],
                'RUT': ['^RUT', 'RUSSELL', 'RUSSELL2000'],
            }
            if symbol in index_map:
                symbol_variants.extend(index_map[symbol])
        
        # Search for matching file
        for filepath in files:
            filename = os.path.basename(filepath).lower()
            for variant in symbol_variants:
                if variant.lower() in filename:
                    return filepath
        
        return None
    
    def _load_and_normalize_csv(self, filepath):
        """
        Load a CSV file and normalize column names
        
        Handles various column naming conventions from different data sources:
        - Kaggle: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
        - CryptoDataDownload: 'date', 'open', 'high', 'low', 'close', 'volume'
        - HistData: '<DTYYYYMMDD>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>'
        """
        try:
            # Try reading CSV
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            else:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"⚠️  Could not read {filepath} with any encoding")
                    return None
            
            # Normalize column names to lowercase
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Map common column name variants
            column_map = {
                # Datetime columns
                'date': 'datetime',
                'time': 'time',
                'timestamp': 'datetime',
                'datetime': 'datetime',
                '<dtyyyymmdd>': 'date',
                '<time>': 'time',
                'unix': 'datetime',
                'unix timestamp': 'datetime',
                
                # OHLCV columns
                'open': 'open',
                '<open>': 'open',
                'price_open': 'open',
                'open_price': 'open',
                
                'high': 'high',
                '<high>': 'high',
                'price_high': 'high',
                'high_price': 'high',
                
                'low': 'low',
                '<low>': 'low',
                'price_low': 'low',
                'low_price': 'low',
                
                'close': 'close',
                '<close>': 'close',
                'price_close': 'close',
                'close_price': 'close',
                'adj close': 'close',
                'adjusted_close': 'close',
                
                'volume': 'volume',
                '<vol>': 'volume',
                'vol': 'volume',
                'volume_traded': 'volume',
            }
            
            # Rename columns
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # Handle datetime
            if 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    # Combine date and time
                    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
                elif 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
                else:
                    # Try first column as datetime
                    df['datetime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Drop rows with invalid datetime
            df = df.dropna(subset=['datetime'])
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Ensure OHLCV columns exist and are numeric
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    print(f"⚠️  Missing column: {col}")
                    return None
            
            if 'volume' in df.columns:
                df[col] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume'] = 0
            
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Drop NaN rows
            df = df.dropna()
            
            # Sort by datetime
            df = df.sort_index()
            
            # Normalize timezone
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            
            return df
        
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            return None
    
    def _get_local_crypto_data(self, symbol, timeframe, max_bars, use_cache):
        """Load crypto data from local Kaggle files"""
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, timeframe)
            if cached_data is not None:
                if max_bars and len(cached_data) > max_bars:
                    return cached_data.tail(max_bars)
                return cached_data
        
        # Find local file
        filepath = self._find_local_file(symbol, 'crypto')
        
        if not filepath:
            return None
        
        print(f"📂 Loading local crypto: {os.path.basename(filepath)}")
        
        # Load and normalize
        df = self._load_and_normalize_csv(filepath)
        
        if df is None or df.empty:
            return None
        
        print(f"   ✅ Loaded {len(df)} bars from local file")
        
        # Resample if needed
        if timeframe != '1min':
            df = self._resample_data(df, timeframe)
            if df is None or df.empty:
                return None
        
        # Cache the result
        self._save_to_cache(symbol, timeframe, df)
        
        # Return requested bars
        if max_bars and len(df) > max_bars:
            return df.tail(max_bars)
        
        return df
    
    def _get_local_indices_data(self, symbol, timeframe, max_bars, use_cache):
        """Load indices data from local Kaggle files"""
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, timeframe)
            if cached_data is not None:
                if max_bars and len(cached_data) > max_bars:
                    return cached_data.tail(max_bars)
                return cached_data
        
        # Find local file
        filepath = self._find_local_file(symbol, 'indices')
        
        if not filepath:
            print(f"❌ No local file found for {symbol}")
            print(f"   Expected in: {config.DATA_CACHE_PATH}/indices/")
            return None
        
        print(f"📂 Loading local index: {os.path.basename(filepath)}")
        
        # Load and normalize
        df = self._load_and_normalize_csv(filepath)
        
        if df is None or df.empty:
            return None
        
        print(f"   ✅ Loaded {len(df)} bars from local file")
        
        # Resample if needed
        if timeframe != '1min' and timeframe != '1day':
            df = self._resample_data(df, timeframe)
            if df is None or df.empty:
                return None
        
        # Cache the result
        self._save_to_cache(symbol, timeframe, df)
        
        # Return requested bars
        if max_bars and len(df) > max_bars:
            return df.tail(max_bars)
        
        return df
    
    # ========================================================================
    # FOREX DATA PIPELINE
    # ========================================================================
    
    def _get_forex_data(self, symbol, timeframe, max_bars, use_cache):
        """
        Load Forex data from local merged CSV files
        
        Process:
        1. Load merged 1-minute base data for this ticker
        2. Resample to requested timeframe if needed
        3. Cache the resampled result
        4. Return requested number of bars
        """
        # Get clean ticker name
        ticker = config.FOREX_TICKERS.get(symbol)
        if not ticker:
            print(f"❌ Unknown Forex ticker: {symbol}")
            print(f"   Add to config.FOREX_TICKERS: '{symbol}': 'TICKER'")
            return None
        
        # Check cache first (for resampled data)
        if use_cache and timeframe != '1min':
            cached_data = self._load_from_cache(symbol, timeframe)
            if cached_data is not None:
                if max_bars and len(cached_data) > max_bars:
                    return cached_data.tail(max_bars)
                return cached_data
        
        # Load base 1-minute merged data
        base_file = os.path.join(config.CACHE_SUBDIRS['forex'], f"{ticker}_1min_merged.csv")
        
        if not os.path.exists(base_file):
            print(f"❌ Missing data: {symbol} (merged file not found)")
            print(f"   Expected: {base_file}")
            print(f"   Run: python forex_data_processor.py first")
            return None
        
        try:
            # Load base data
            df = pd.read_csv(base_file, index_col=0, parse_dates=True)
            
            # Normalize timezone
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            
            # If requesting 1-minute data, return it directly
            if timeframe == '1min':
                if max_bars and len(df) > max_bars:
                    return df.tail(max_bars)
                return df
            
            # Resample to higher timeframe
            resampled = self._resample_data(df, timeframe)
            
            if resampled is None or resampled.empty:
                print(f"❌ Missing data: {symbol} {timeframe} (resampling failed)")
                return None
            
            # Save resampled data to cache
            self._save_to_cache(symbol, timeframe, resampled)
            
            # Return requested bars
            if max_bars and len(resampled) > max_bars:
                return resampled.tail(max_bars)
            
            return resampled
        
        except Exception as e:
            print(f"❌ Missing data: {symbol} {timeframe}")
            print(f"   Error: {e}")
            return None
    
    def _resample_data(self, df, timeframe):
        """
        Resample OHLCV data to a higher timeframe
        
        OHLCV Aggregation Rules:
        - Open: first
        - High: max
        - Low: min
        - Close: last
        - Volume: sum
        """
        # Timeframe mapping to pandas resample rule
        resample_rules = {
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1hour': '1H',
            '4hour': '4H',
            '1day': '1D',
            '1week': '1W',
        }
        
        rule = resample_rules.get(timeframe)
        if not rule:
            print(f"⚠️  Unknown timeframe for resampling: {timeframe}")
            return None
        
        # Build aggregation dictionary
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        
        # Add volume if it exists
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        
        # Resample and drop NaN rows
        resampled = df.resample(rule).agg(agg_dict).dropna()
        
        return resampled
    
    # ========================================================================
    # CRYPTO DATA PIPELINE (CCXT FALLBACK)
    # ========================================================================
    
    def _get_crypto_data_ccxt(self, symbol, timeframe, max_bars, use_cache):
        """
        Fetch crypto data via CCXT (fallback when no local files)
        
        Priority: Binance → Kraken → Hyperliquid
        """
        # Check cache first
        if use_cache:
            for exchange_name in config.CRYPTO_EXCHANGE_PRIORITY:
                cached_data = self._load_from_cache(symbol, timeframe, exchange=exchange_name)
                if cached_data is not None:
                    if max_bars and len(cached_data) > max_bars:
                        return cached_data.tail(max_bars)
                    return cached_data
        
        # Get exchange-specific symbol
        symbol_map = config.CRYPTO_SYMBOL_MAP.get(symbol, {})
        
        # Try exchanges in priority order, plus Kraken as backup
        exchanges_to_try = list(config.CRYPTO_EXCHANGE_PRIORITY) + ['kraken']
        
        for exchange_name in exchanges_to_try:
            if exchange_name not in self.exchanges:
                continue
            
            # Get symbol for this exchange
            if exchange_name == 'kraken':
                ccxt_symbol = symbol.replace('-', '/')
            elif exchange_name in symbol_map:
                ccxt_symbol = symbol_map.get(exchange_name)
            else:
                ccxt_symbol = symbol.replace('-USD', '/USDT')
            
            print(f"📊 Fetching {symbol} ({timeframe}) from {exchange_name.upper()}...")
            
            try:
                exchange = self.exchanges[exchange_name]
                data = self._fetch_ccxt(exchange, ccxt_symbol, timeframe, max_bars)
                
                if data is not None and not data.empty:
                    print(f"  ✅ Retrieved {len(data)} bars from {exchange_name.upper()}")
                    self._save_to_cache(symbol, timeframe, data, exchange=exchange_name)
                    return data
                else:
                    print(f"  ℹ️  No data from {exchange_name.upper()}, trying next...")
            
            except Exception as e:
                print(f"  ❌ Error from {exchange_name.upper()}: {str(e)[:100]}")
                continue
        
        print(f"❌ Missing data: {symbol} {timeframe}")
        print(f"   Tried: {', '.join(config.CRYPTO_EXCHANGE_PRIORITY)}")
        return None
    
    def _fetch_ccxt(self, exchange, symbol, timeframe, max_bars=None):
        """
        Fetch OHLCV data from a CCXT exchange
        """
        ccxt_timeframe = config.TIMEFRAMES.get(timeframe, '1h')
        limit = max_bars if max_bars else 1000
        
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            raise e
    
    # ========================================================================
    # CACHE SYSTEM
    # ========================================================================
    
    def _get_cache_filename(self, symbol, timeframe, exchange=None):
        """Generate cache filename for a symbol/timeframe combination"""
        asset_type = self._determine_asset_type(symbol)
        
        if asset_type in config.CACHE_SUBDIRS:
            subdir = config.CACHE_SUBDIRS[asset_type]
        else:
            subdir = config.CACHE_SUBDIRS.get('raw', config.CACHE_SUBDIRS['forex'])
        
        clean_symbol = symbol.replace('=', '').replace('^', '').replace('-', '').replace('/', '')
        
        if exchange:
            filename = f"{clean_symbol}_{timeframe}_{exchange}.csv"
        else:
            filename = f"{clean_symbol}_{timeframe}.csv"
        
        return os.path.join(subdir, filename)
    
    def _load_from_cache(self, symbol, timeframe, exchange=None):
        """Load data from cache file"""
        cache_file = self._get_cache_filename(symbol, timeframe, exchange)
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, symbol, timeframe, data, exchange=None):
        """Save data to cache file"""
        cache_file = self._get_cache_filename(symbol, timeframe, exchange)
        
        try:
            data.to_csv(cache_file)
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
    
    def clear_cache(self, symbol=None, timeframe=None):
        """Clear cached data"""
        if symbol and timeframe:
            cache_file = self._get_cache_filename(symbol, timeframe)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Cleared cache for {symbol} {timeframe}")
        else:
            cleared = 0
            for subdir in config.CACHE_SUBDIRS.values():
                if os.path.exists(subdir):
                    for file in os.listdir(subdir):
                        if file.endswith('.csv') and 'merged' not in file:
                            os.remove(os.path.join(subdir, file))
                            cleared += 1
            print(f"🗑️  Cleared {cleared} cache files")


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def get_data(symbol, timeframe='1hour', max_bars=None, **kwargs):
    """
    Quick data fetch (creates DataManager instance automatically)
    
    Usage:
        from data_manager import get_data
        data = get_data('BTC-USD', '1hour', 1000)
    """
    manager = DataManager()
    return manager.get_data(symbol, timeframe, max_bars, **kwargs)


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DATA MANAGER TEST")
    print("="*70)
    
    manager = DataManager()
    
    # Test Crypto (local files first)
    if config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        print(f"\n📊 Testing Crypto (local files → CCXT fallback):")
        for symbol in config.CRYPTO_WATCHLIST[:2]:  # Test first 2
            data = manager.get_data(symbol, '1hour', 100)
            if data is not None:
                print(f"   ✅ {symbol}: {len(data)} bars")
            else:
                print(f"   ❌ {symbol}: No data")
    
    # Test Indices (local files)
    if config.INDICES_ENABLED and config.INDEX_WATCHLIST:
        print(f"\n📊 Testing Indices (local files):")
        for symbol in config.INDEX_WATCHLIST[:2]:  # Test first 2
            data = manager.get_data(symbol, '1day', 100)
            if data is not None:
                print(f"   ✅ {symbol}: {len(data)} bars")
            else:
                print(f"   ❌ {symbol}: No data")
    
    # Test Forex
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        print(f"\n📊 Testing Forex:")
        data = manager.get_data(config.FOREX_WATCHLIST[0], '1hour', 100)
        if data is not None:
            print(f"   ✅ {config.FOREX_WATCHLIST[0]}: {len(data)} bars")
        else:
            print(f"   ❌ {config.FOREX_WATCHLIST[0]}: No data")
    
    print("\n" + "="*70)