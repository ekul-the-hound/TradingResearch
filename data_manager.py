# ==============================================================================
# data_manager.py
# ==============================================================================
# Unified data manager for the Trading Research system
#
# Data Sources:
# - Forex: Local CSV files only (E:/TradingData/forex)
# - Crypto: CCXT only (Binance → Hyperliquid fallback)
# - Indices: DISABLED (awaiting IBKR integration)
# - Commodities: DISABLED (awaiting futures data source)
#
# Features:
# - Automatic resampling of 1-minute Forex data to higher timeframes
# - CCXT-based crypto data fetching with exchange fallback
# - Caching system for performance
# - Graceful handling of disabled asset classes
# ==============================================================================

import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
import config

# CCXT for crypto data
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
    """
    
    def __init__(self):
        self._ensure_cache_dirs()
        self.exchanges = {}
        
        # Initialize CCXT exchanges if available and crypto enabled
        if CCXT_AVAILABLE and config.CRYPTO_ENABLED:
            self._init_exchanges()
    
    def _ensure_cache_dirs(self):
        """Create cache directories if they don't exist"""
        for subdir in config.CACHE_SUBDIRS.values():
            Path(subdir).mkdir(parents=True, exist_ok=True)
    
    def _init_exchanges(self):
        """Initialize CCXT exchanges"""
        self.exchanges = {}
        
        if config.BINANCE_ENABLED and 'binance' in config.CRYPTO_EXCHANGE_PRIORITY:
            try:
                exchange_config = {
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
                # Add API keys if available
                if config.BINANCE_API_KEY:
                    exchange_config['apiKey'] = config.BINANCE_API_KEY
                    exchange_config['secret'] = config.BINANCE_API_SECRET
                
                # Use binanceus for US users (binance.com blocked in USA)
                self.exchanges['binance'] = ccxt.binanceus(exchange_config)
                print("✅ Binance US exchange initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize Binance US: {e}")
        
        # Also try Kraken as a reliable US-friendly backup
        try:
            self.exchanges['kraken'] = ccxt.kraken({'enableRateLimit': True})
            print("✅ Kraken exchange initialized")
        except Exception as e:
            print(f"⚠️  Failed to initialize Kraken: {e}")
        
        if config.HYPERLIQUID_ENABLED and 'hyperliquid' in config.CRYPTO_EXCHANGE_PRIORITY:
            try:
                self.exchanges['hyperliquid'] = ccxt.hyperliquid({
                    'enableRateLimit': True
                })
                print("✅ Hyperliquid exchange initialized")
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
        elif symbol.endswith('-USD') or symbol.endswith('USDT'):
            return 'crypto'
        elif symbol.startswith('^'):
            return 'indices'
        elif symbol.endswith('=F'):
            return 'commodities'
        
        return 'unknown'
    
    def get_data(self, symbol, timeframe='1hour', max_bars=None, use_cache=True, **kwargs):
        """
        Get OHLCV data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EUR-USD', 'BTC-USD')
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
        # INDICES: DISABLED
        # ====================================================================
        if asset_type == 'indices':
            if not config.INDICES_ENABLED:
                print(f"⏸️  Data not added yet: {symbol} {timeframe}")
                print(f"   (Indices awaiting IBKR integration)")
                return None
        
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
        # CRYPTO: CCXT only
        # ====================================================================
        if asset_type == 'crypto':
            if not config.CRYPTO_ENABLED:
                print(f"⏸️  Crypto disabled in config")
                return None
            
            if not CCXT_AVAILABLE:
                print(f"❌ CCXT not installed. Run: pip install ccxt")
                return None
            
            if not self.exchanges:
                print(f"❌ No exchanges initialized for {symbol}")
                return None
            
            return self._get_crypto_data(symbol, timeframe, max_bars, use_cache)
        
        # ====================================================================
        # UNKNOWN ASSET TYPE
        # ====================================================================
        print(f"❌ Unknown asset type for {symbol}")
        return None
    
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
            resampled = self._resample_forex(df, timeframe)
            
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
    
    def _resample_forex(self, df, timeframe):
        """
        Resample 1-minute Forex data to higher timeframe
        
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
            '1day': '1D'
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
    # CRYPTO DATA PIPELINE (CCXT)
    # ========================================================================
    
    def _get_crypto_data(self, symbol, timeframe, max_bars, use_cache):
        """
        Fetch crypto data via CCXT
        
        Priority: Binance → Hyperliquid
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
            
            # Get symbol for this exchange - handle different formats
            if exchange_name == 'kraken':
                # Kraken uses BTC/USD format (actual USD, not USDT)
                ccxt_symbol = symbol.replace('-', '/')
            elif exchange_name in symbol_map:
                ccxt_symbol = symbol_map.get(exchange_name)
            else:
                # Default: convert BTC-USD to BTC/USDT
                ccxt_symbol = symbol.replace('-USD', '/USDT')
            
            print(f"📊 Fetching {symbol} ({timeframe}) from {exchange_name.upper()}...")
            
            try:
                exchange = self.exchanges[exchange_name]
                data = self._fetch_ccxt(exchange, ccxt_symbol, timeframe, max_bars)
                
                if data is not None and not data.empty:
                    print(f"  ✅ Retrieved {len(data)} bars from {exchange_name.upper()}")
                    
                    # Save to cache
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
        
        Args:
            exchange: CCXT exchange instance
            symbol: Exchange-specific symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string
            max_bars: Maximum bars to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        # Convert timeframe to CCXT format
        ccxt_timeframe = config.TIMEFRAMES.get(timeframe, '1h')
        
        # Determine limit
        limit = max_bars if max_bars else 1000
        
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
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
        
        # Get appropriate cache subdirectory
        if asset_type in config.CACHE_SUBDIRS:
            subdir = config.CACHE_SUBDIRS[asset_type]
        else:
            subdir = config.CACHE_SUBDIRS.get('raw', config.CACHE_SUBDIRS['forex'])
        
        # Clean symbol for filename
        clean_symbol = symbol.replace('=', '').replace('^', '').replace('-', '').replace('/', '')
        
        # Add exchange suffix for crypto
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
        """
        Clear cached data
        
        Args:
            symbol: Specific symbol to clear (None = all)
            timeframe: Specific timeframe to clear (None = all)
        """
        if symbol and timeframe:
            # Clear specific file
            cache_file = self._get_cache_filename(symbol, timeframe)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Cleared cache for {symbol} {timeframe}")
        else:
            # Clear all non-merged cache files
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
    print("DATA MANAGER QUICK TEST")
    print("="*70)
    
    manager = DataManager()
    
    # Test Forex
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        print(f"\nTesting Forex: {config.FOREX_WATCHLIST[0]}")
        data = manager.get_data(config.FOREX_WATCHLIST[0], '1hour', 100)
        if data is not None:
            print(f"  ✅ Got {len(data)} bars")
        else:
            print(f"  ❌ Failed")
    
    # Test Crypto
    if config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        print(f"\nTesting Crypto: {config.CRYPTO_WATCHLIST[0]}")
        data = manager.get_data(config.CRYPTO_WATCHLIST[0], '1hour', 100)
        if data is not None:
            print(f"  ✅ Got {len(data)} bars")
        else:
            print(f"  ❌ Failed")
    
    print("\n" + "="*70)