# ==============================================================================
# backtest_adapter.py
# ==============================================================================
# Bridges the gap between:
#   - Parameter vectors (what the optimizer produces)
#   - Backtrader strategy classes (what your backtester consumes)
#   - CanonicalResult objects (what all downstream modules consume)
#
# Three modes of operation:
#   1. evaluate_params()   — GA proposes parameter vector → backtest → result
#   2. evaluate_strategy() — Run existing strategy class → result
#   3. evaluate_variant()  — Load .py variant file → backtest → result
#
# This is the architectural adapter that closes the loop.
#
# Usage:
#     from backtest_adapter import BacktestAdapter
#     adapter = BacktestAdapter()
#
#     # Mode 1: GA optimization loop
#     result = adapter.evaluate_params(
#         param_vector={"fast_period": 10, "slow_period": 50},
#         strategy_class=SimpleMovingAverageCrossover,
#         symbol="EUR-USD", timeframe="1hour"
#     )
#
#     # Mode 2: Run existing strategy
#     result = adapter.evaluate_strategy(
#         strategy_class=MyStrategy,
#         symbols=["EUR-USD", "GBP-USD"],
#         timeframes=["1hour"]
#     )
#
#     # Mode 3: Load and run a .py variant file
#     result = adapter.evaluate_variant("strategies/variants/variant_001.py")
# ==============================================================================

import importlib
import importlib.util
import sys
import traceback
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Type

# Import from your existing system
sys.path.insert(0, str(Path(__file__).parent))

try:
    from backtester_multi_timeframe import MultiTimeframeBacktester
except ImportError:
    MultiTimeframeBacktester = None

try:
    from backtester import StrategyBacktester
except ImportError:
    StrategyBacktester = None

from canonical_result import CanonicalResult


class BacktestAdapter:
    """
    Adapter that connects optimizers/pipelines to your existing backtester.
    """

    def __init__(
        self,
        default_symbols: Optional[List[str]] = None,
        default_timeframes: Optional[List[str]] = None,
        default_initial_cash: float = 10000,
        default_commission: float = 0.001,
        extract_trades: bool = True,
        verbose: bool = True,
    ):
        self.default_symbols = default_symbols or ["EUR-USD"]
        self.default_timeframes = default_timeframes or ["1hour"]
        self.default_initial_cash = default_initial_cash
        self.default_commission = default_commission
        self.extract_trades = extract_trades
        self.verbose = verbose

        # Initialize backtester
        self._mtf = None
        self._simple = None
        self._init_backtester()

        self._eval_count = 0

    def _init_backtester(self):
        """Initialize whichever backtester is available."""
        if MultiTimeframeBacktester is not None:
            try:
                self._mtf = MultiTimeframeBacktester()
                if self.verbose:
                    print("✅ BacktestAdapter: Using MultiTimeframeBacktester")
                return
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  MultiTimeframeBacktester init failed: {e}")

        if StrategyBacktester is not None:
            try:
                self._simple = StrategyBacktester()
                if self.verbose:
                    print("✅ BacktestAdapter: Using StrategyBacktester")
                return
            except Exception:
                pass

        if self.verbose:
            print("⚠️  BacktestAdapter: No backtester available (dry-run mode)")

    # ------------------------------------------------------------------
    # MODE 1: Evaluate a parameter vector (for GA/optimizer)
    # ------------------------------------------------------------------
    def evaluate_params(
        self,
        param_vector: Dict[str, Any],
        strategy_class: Type,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> CanonicalResult:
        """
        Core adapter for optimization loops.

        Takes a parameter dict, injects it into the strategy class,
        runs your backtester, returns a CanonicalResult.

        Args:
            param_vector: Dict of parameter names → values.
                          e.g. {"fast_period": 10, "slow_period": 50}
            strategy_class: Backtrader strategy class (e.g. SimpleMovingAverageCrossover)
            symbol: Asset to test (defaults to first in default_symbols)
            timeframe: Timeframe to test (defaults to first in default_timeframes)
            strategy_id: Optional ID for tracking

        Returns:
            CanonicalResult with all metrics populated.
        """
        self._eval_count += 1
        sym = symbol or self.default_symbols[0]
        tf = timeframe or self.default_timeframes[0]

        if not strategy_id:
            params_str = "_".join(f"{k}{v}" for k, v in sorted(param_vector.items()))
            strategy_id = f"{strategy_class.__name__}_{params_str}_{sym}_{tf}"

        raw = self._run_backtest(strategy_class, sym, tf, param_vector)

        cr = CanonicalResult.from_backtest(raw, strategy_id=strategy_id)
        cr.strategy_params = dict(param_vector)
        return cr

    # ------------------------------------------------------------------
    # MODE 1b: Batch evaluate for GA (run across multiple symbols)
    # ------------------------------------------------------------------
    def evaluate_params_multi(
        self,
        param_vector: Dict[str, Any],
        strategy_class: Type,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        aggregate: str = "mean",
    ) -> CanonicalResult:
        """
        Evaluate a parameter vector across multiple symbols/timeframes.
        Returns aggregated CanonicalResult.
        """
        syms = symbols or self.default_symbols
        tfs = timeframes or self.default_timeframes

        results = []
        for sym in syms:
            for tf in tfs:
                cr = self.evaluate_params(param_vector, strategy_class, sym, tf)
                if cr.total_trades > 0:
                    results.append(cr)

        if not results:
            return CanonicalResult(strategy_id="EMPTY", strategy_params=param_vector)

        # Aggregate
        return self._aggregate_results(results, param_vector, strategy_class.__name__)

    # ------------------------------------------------------------------
    # MODE 2: Evaluate an existing strategy class
    # ------------------------------------------------------------------
    def evaluate_strategy(
        self,
        strategy_class: Type,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        strategy_params: Optional[Dict] = None,
        strategy_id: Optional[str] = None,
    ) -> List[CanonicalResult]:
        """
        Run a strategy class across symbols/timeframes.
        Returns list of CanonicalResult, one per symbol/timeframe combo.
        """
        syms = symbols or self.default_symbols
        tfs = timeframes or self.default_timeframes
        params = strategy_params or {}

        results = []
        for sym in syms:
            for tf in tfs:
                sid = strategy_id or f"{strategy_class.__name__}_{sym}_{tf}"
                raw = self._run_backtest(strategy_class, sym, tf, params)
                cr = CanonicalResult.from_backtest(raw, strategy_id=sid)
                results.append(cr)

        return results

    # ------------------------------------------------------------------
    # MODE 3: Evaluate a .py variant file
    # ------------------------------------------------------------------
    def evaluate_variant(
        self,
        variant_path: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> CanonicalResult:
        """
        Load a strategy .py file (e.g., from mutation agent), import the
        strategy class, and run it.
        """
        path = Path(variant_path)
        if not path.exists():
            return CanonicalResult(strategy_id=path.stem, strategy_name="FILE_NOT_FOUND")

        # Dynamic import
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to import {path}: {e}")
            return CanonicalResult(strategy_id=path.stem, strategy_name="IMPORT_FAILED")

        # Find the strategy class (first bt.Strategy subclass)
        import backtrader as bt
        strategy_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, bt.Strategy) and attr is not bt.Strategy:
                strategy_class = attr
                break

        if strategy_class is None:
            return CanonicalResult(strategy_id=path.stem, strategy_name="NO_STRATEGY_CLASS")

        sym = symbol or self.default_symbols[0]
        tf = timeframe or self.default_timeframes[0]

        raw = self._run_backtest(strategy_class, sym, tf, {})
        cr = CanonicalResult.from_backtest(raw, strategy_id=path.stem)
        return cr

    # ------------------------------------------------------------------
    # MODE 4: Callable for optimization_pipeline.py
    # ------------------------------------------------------------------
    def as_objective_function(
        self,
        strategy_class: Type,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        """
        Returns a callable that optimization_pipeline.py can use directly.

        Usage:
            obj_fn = adapter.as_objective_function(SimpleMovingAverageCrossover)
            # optimization_pipeline.py calls: obj_fn(param_dict) → result_dict
        """
        def _objective(param_vector: Dict[str, Any]) -> Dict[str, float]:
            cr = self.evaluate_params(param_vector, strategy_class, symbol, timeframe)
            return {
                "sharpe_ratio": cr.sharpe_ratio or 0,
                "max_drawdown_pct": cr.max_drawdown_pct or 0,
                "total_trades": float(cr.total_trades),
                "win_rate": (cr.win_rate or 50) / 100,
                "profit_factor": cr.profit_factor or 0,
                "total_return_pct": cr.total_return_pct,
            }
        return _objective

    # ------------------------------------------------------------------
    # INTERNAL: Run backtest through your existing engine
    # ------------------------------------------------------------------
    def _run_backtest(
        self,
        strategy_class: Type,
        symbol: str,
        timeframe: str,
        strategy_params: Dict,
    ) -> Optional[Dict]:
        """Run backtest using whichever engine is available."""

        # Multi-timeframe backtester (preferred)
        if self._mtf is not None:
            try:
                result = self._mtf.run_single_backtest(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_cash=self.default_initial_cash,
                    commission=self.default_commission,
                    strategy_params=strategy_params if strategy_params else None,
                    extract_trades=self.extract_trades,
                )
                return result
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Backtest failed ({symbol} {timeframe}): {e}")
                return None

        # Simple backtester fallback (uses yfinance, no timeframe/forex support)
        if self._simple is not None:
            try:
                result = self._simple.run_backtest(
                    strategy_class=strategy_class,
                    symbol=symbol.replace("-", "=X") if "-" in symbol else symbol,
                    start_date="2020-01-01",
                    end_date="2025-01-01",
                    initial_cash=self.default_initial_cash,
                    commission=self.default_commission,
                    strategy_params=strategy_params if strategy_params else None,
                )
                return result
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Simple backtest failed: {e}")
                return None

        # No backtester — return None (CanonicalResult handles gracefully)
        return None

    # ------------------------------------------------------------------
    # AGGREGATION
    # ------------------------------------------------------------------
    def _aggregate_results(
        self,
        results: List[CanonicalResult],
        params: Dict,
        name: str,
    ) -> CanonicalResult:
        """Aggregate multiple single-asset results into one."""
        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio]
        dds = [r.max_drawdown_pct for r in results if r.max_drawdown_pct]
        trades = [r.total_trades for r in results]
        returns = [r.total_return_pct for r in results]
        wrs = [r.win_rate for r in results if r.win_rate is not None]
        pfs = [r.profit_factor for r in results if r.profit_factor is not None]

        # Concatenate return arrays
        all_returns = np.concatenate([r.returns for r in results if r.returns is not None])

        agg = CanonicalResult(
            strategy_id=f"{name}_agg",
            strategy_name=name,
            strategy_params=params,
            sharpe_ratio=float(np.mean(sharpes)) if sharpes else 0,
            max_drawdown_pct=float(np.max(dds)) if dds else 0,
            total_trades=sum(trades),
            total_return_pct=float(np.mean(returns)),
            win_rate=float(np.mean(wrs)) if wrs else None,
            profit_factor=float(np.mean(pfs)) if pfs else None,
            returns=all_returns,
        )
        return agg

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------
    @property
    def eval_count(self) -> int:
        return self._eval_count

    def reset_count(self):
        self._eval_count = 0
