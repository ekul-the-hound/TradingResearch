# ==============================================================================
# canonical_result.py
# ==============================================================================
# The ONE data structure that every TradingLab module consumes.
#
# Your backtester produces a dict. This module wraps it into a standard
# object with computed fields (daily returns, equity curve, etc.) that
# downstream modules need. No module should ever parse raw backtest
# output directly -- they all go through CanonicalResult.
#
# Consumed by: every Phase 1-6 module.
#
# Usage:
#     from canonical_result import CanonicalResult
#     cr = CanonicalResult.from_backtest(backtest_result_dict)
#     cr.returns          # np.ndarray of daily returns
#     cr.equity_curve     # np.ndarray of equity values
#     cr.sharpe           # float
#     cr.to_phase1_dict() # dict for filtering_pipeline
#     cr.to_phase3_dict() # dict for market_impact, capacity_model
# ==============================================================================

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CanonicalResult:
    """
    Standard result object for the entire TradingLab pipeline.
    Created from your backtester output, consumed by all downstream modules.
    """

    # -- Identity ----------------------------------------------------------
    strategy_id: str = ""
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # -- Core metrics (from backtester) ------------------------------------
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate: Optional[float] = None       # percentage (0-100)
    profit_factor: Optional[float] = None

    # -- Extended metrics --------------------------------------------------
    starting_value: float = 10000.0
    ending_value: float = 10000.0
    bars_tested: int = 0
    start_date: str = ""
    end_date: str = ""
    trades_per_day: float = 0.0
    avg_trade_return_pct: float = 0.0
    avg_trade_duration_bars: float = 0.0
    time_in_market_pct: float = 0.0

    # -- Computed arrays (built from trade list / equity) ------------------
    returns: Optional[np.ndarray] = None          # daily returns (decimal)
    equity_curve: Optional[np.ndarray] = None     # equity values over time
    trade_list: List[Dict] = field(default_factory=list)

    # -- Lineage -----------------------------------------------------------
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    hypothesis_id: Optional[str] = None
    generation: int = 0

    # ------------------------------------------------------------------
    # CONSTRUCTORS
    # ------------------------------------------------------------------
    @classmethod
    def from_backtest(cls, result: Dict[str, Any], strategy_id: str = "") -> "CanonicalResult":
        """
        Build from your backtester's result dict.
        Works with both backtester.py and backtester_multi_timeframe.py output.
        """
        if result is None:
            return cls(strategy_id=strategy_id or "FAILED")

        # Auto-generate strategy_id if not provided
        if not strategy_id:
            name = result.get("strategy_name", "unknown")
            sym = result.get("symbol", "")
            tf = result.get("timeframe", "")
            strategy_id = f"{name}_{sym}_{tf}".replace(" ", "_")

        cr = cls(
            strategy_id=strategy_id,
            strategy_name=result.get("strategy_name", ""),
            symbol=result.get("symbol", ""),
            timeframe=result.get("timeframe", ""),
            strategy_params=result.get("strategy_params", {}),
            total_return_pct=result.get("total_return_pct", 0) or 0,
            sharpe_ratio=result.get("sharpe_ratio") or 0,
            max_drawdown_pct=result.get("max_drawdown_pct", 0) or 0,
            total_trades=result.get("total_trades", 0) or 0,
            win_rate=result.get("win_rate"),
            profit_factor=result.get("profit_factor"),
            starting_value=result.get("starting_value", 10000),
            ending_value=result.get("ending_value", 10000),
            bars_tested=result.get("bars_tested", 0) or 0,
            start_date=result.get("start_date", ""),
            end_date=result.get("end_date", ""),
            trades_per_day=result.get("trades_per_day", 0) or 0,
            avg_trade_return_pct=result.get("avg_trade_return_pct", 0) or 0,
            avg_trade_duration_bars=result.get("avg_trade_duration_bars", 0) or 0,
            time_in_market_pct=result.get("time_in_market_pct", 0) or 0,
        )

        # Extract trade list if present
        if "trades" in result and result["trades"]:
            cr.trade_list = list(result["trades"])

        # Build returns and equity curve from trade list
        cr._compute_arrays()

        return cr

    def _compute_arrays(self):
        """Compute daily returns and equity curve from available data."""
        if self.trade_list:
            # Build equity curve from sequential trade PnLs
            equity = [self.starting_value]
            for t in self.trade_list:
                pnl = t.get("pnl", t.get("profit", 0)) or 0
                equity.append(equity[-1] + pnl)
            self.equity_curve = np.array(equity, dtype=np.float64)

            # Compute returns from equity curve
            eq = self.equity_curve
            if len(eq) > 1:
                self.returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
            else:
                self.returns = np.array([0.0])

        elif self.total_return_pct != 0 and self.bars_tested > 0:
            # Synthesize approximate daily returns from summary stats
            n = max(self.bars_tested, 1)
            total_r = self.total_return_pct / 100
            daily_r = (1 + total_r) ** (1 / n) - 1

            # Add noise scaled to match Sharpe if available
            if self.sharpe_ratio and self.sharpe_ratio != 0:
                daily_vol = abs(daily_r) / max(abs(self.sharpe_ratio / np.sqrt(252)), 1e-6)
            else:
                daily_vol = abs(daily_r) * 2

            rng = np.random.RandomState(hash(self.strategy_id) % 2**31)
            self.returns = rng.normal(daily_r, max(daily_vol, 1e-8), n)

            # Build equity curve
            equity = self.starting_value * np.cumprod(1 + self.returns)
            self.equity_curve = np.concatenate([[self.starting_value], equity])
        else:
            self.returns = np.array([0.0])
            self.equity_curve = np.array([self.starting_value])

    # ------------------------------------------------------------------
    # OUTPUT FORMATS (one per consumer)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Full dict representation."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_params": self.strategy_params,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "starting_value": self.starting_value,
            "ending_value": self.ending_value,
            "bars_tested": self.bars_tested,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "avg_trade_pct": self.avg_trade_return_pct,
        }

    def to_filter_dict(self) -> Dict[str, Any]:
        """Format for filtering_pipeline.py and overfitting_detector.py."""
        return {
            "strategy_id": self.strategy_id,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_return_pct": self.total_return_pct,
        }

    def to_risk_dict(self) -> Dict[str, Any]:
        """Format for market_impact.py, capacity_model.py, tail_risk.py."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "total_return_pct": self.total_return_pct,
            "avg_trade_pct": self.avg_trade_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
        }

    def to_fingerprint_input(self) -> Dict[str, float]:
        """Format for strategy_fingerprint.py."""
        return {
            "sharpe_ratio": self.sharpe_ratio or 0,
            "max_drawdown_pct": self.max_drawdown_pct or 0,
            "total_trades": float(self.total_trades),
            "win_rate": (self.win_rate or 50) / 100,
            "profit_factor": self.profit_factor or 1.0,
            "total_return_pct": self.total_return_pct,
            "trades_per_day": self.trades_per_day,
            "avg_trade_return_pct": self.avg_trade_return_pct,
            "time_in_market_pct": self.time_in_market_pct,
        }

    def to_lineage_dict(self) -> Dict[str, Any]:
        """Format for lineage_tracker.py and lineage_analytics.py."""
        return {
            "strategy_id": self.strategy_id,
            "parent_id": self.parent_id,
            "mutation_type": self.mutation_type,
            "hypothesis_id": self.hypothesis_id,
            "generation": self.generation,
            "backtest_sharpe": self.sharpe_ratio,
            "live_sharpe": 0,  # Populated later during shadow trading
            "max_drawdown": self.max_drawdown_pct / 100 if self.max_drawdown_pct else 0,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor or 0,
        }

    def __str__(self):
        sr = f"{self.sharpe_ratio:.3f}" if self.sharpe_ratio else "N/A"
        wr = f"{self.win_rate:.1f}%" if self.win_rate else "N/A"
        return (
            f"[{self.strategy_id}] {self.symbol} {self.timeframe} | "
            f"Return: {self.total_return_pct:+.2f}% | Sharpe: {sr} | "
            f"DD: {self.max_drawdown_pct:.1f}% | Trades: {self.total_trades} | WR: {wr}"
        )
