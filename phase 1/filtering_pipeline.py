# ==============================================================================
# filtering_pipeline.py
# ==============================================================================
# Module 3 of 4 — Phase 1: Foundation Completion
#
# Automated Strategy Filtering Pipeline
#
# Chains: backtest results → PBO/DSR check → hard threshold filters →
#         weighted composite score → rank → select top N survivors
#
# No external GitHub repos — pure orchestration logic consuming:
#   - lineage_tracker.py (Module 1) for status updates
#   - overfitting_detector.py (Module 2) for PBO + DSR scoring
#   - Existing backtest results from database.py
#
# Consumed by:
#   - phase1_pipeline.py (integration orchestrator)
#   - diversification_filter.py (Module 4) receives survivors
#
# Usage:
#     from filtering_pipeline import FilteringPipeline, FilterConfig
#
#     pipeline = FilteringPipeline()
#     result = pipeline.run(
#         strategies=all_strategies,
#         config=FilterConfig(min_sharpe=0.5, max_drawdown=25.0, min_trades=50)
#     )
#     print(result.summary())
#
# ==============================================================================

import numpy as np
import pandas as pd
import sqlite3
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from lineage_tracker import LineageTracker
except ImportError:
    LineageTracker = None

try:
    from overfitting_detector import OverfittingDetector
except ImportError:
    OverfittingDetector = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class FilterConfig:
    """All thresholds and weights for the pipeline."""

    # -- Hard filters (fail = eliminated) --
    min_sharpe: float = 0.3
    max_drawdown: float = 30.0           # %
    min_trades: int = 30
    min_win_rate: float = 0.0            # 0 = disabled
    min_profit_factor: float = 0.0       # 0 = disabled
    max_pbo: float = 0.50               # reject if PBO > this
    min_total_return: float = -100.0     # %

    # -- Composite score weights --
    weight_sharpe: float = 1.0
    weight_drawdown: float = 0.8
    weight_profit_factor: float = 0.6
    weight_win_rate: float = 0.3
    weight_trades: float = 0.2
    weight_pbo: float = 0.5
    weight_dsr: float = 0.4

    # -- Selection --
    top_n: int = 100
    top_pct: Optional[float] = None      # if set, overrides top_n


# ==============================================================================
# RESULT DATACLASSES
# ==============================================================================

@dataclass
class FilterResult:
    """Result for one strategy through the pipeline."""
    strategy_id: str
    name: str
    passed: bool
    rejection_reasons: List[str]
    composite_score: float
    rank: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    overfitting_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Overall pipeline output."""
    total_input: int
    total_passed_hard: int
    total_survivors: int
    survivors: List[FilterResult]
    rejected: List[FilterResult]
    config: FilterConfig
    timestamp: str = ""
    filter_stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  FILTERING PIPELINE RESULTS",
            f"{'='*60}",
            f"  Input strategies:    {self.total_input}",
            f"  Passed hard filters: {self.total_passed_hard}",
            f"  Final survivors:     {self.total_survivors}",
            f"{'='*60}",
        ]
        if self.filter_stats:
            lines.append("  Rejection breakdown:")
            for r, c in sorted(self.filter_stats.items(), key=lambda x: -x[1]):
                lines.append(f"    {r}: {c}")
        if self.survivors:
            lines.append(f"\n  Top 5 survivors:")
            for s in self.survivors[:5]:
                sr = s.metrics.get("sharpe_ratio", 0) or 0
                lines.append(f"    #{s.rank}: {s.name} "
                             f"(score={s.composite_score:.4f}, Sharpe={sr:.3f})")
        return "\n".join(lines)


# ==============================================================================
# FILTERING PIPELINE
# ==============================================================================

class FilteringPipeline:
    """
    Applies hard filters, scores, ranks, selects top N.
    """

    def __init__(
        self,
        lineage_tracker: Optional[Any] = None,
        overfitting_detector: Optional[Any] = None,
    ):
        self.lineage = lineage_tracker
        self.detector = overfitting_detector or (
            OverfittingDetector() if OverfittingDetector else None
        )

    # ------------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------------
    def run(
        self,
        strategies: List[Dict[str, Any]],
        config: Optional[FilterConfig] = None,
        returns_matrix: Optional[pd.DataFrame] = None,
    ) -> PipelineResult:
        """
        Run the full filtering pipeline.

        Args:
            strategies: List of dicts with at minimum:
                strategy_id, name, sharpe_ratio, max_drawdown_pct,
                total_trades, total_return_pct.
                Optional: win_rate, profit_factor, returns (np.array).
            config: Thresholds and weights.
            returns_matrix: Optional (T, N) DataFrame for PBO analysis.
        """
        if config is None:
            config = FilterConfig()

        print(f"\n🔬 Filtering Pipeline: {len(strategies)} strategies → top {config.top_n}")

        # Step 1: PBO if returns matrix provided
        pbo_scores: Dict[str, float] = {}
        if returns_matrix is not None and self.detector is not None:
            print("  📊 Computing PBO...")
            pbo_scores = self._compute_pbo_scores(returns_matrix)

        # Step 2: Apply hard filters + score
        results: List[FilterResult] = []
        filter_stats: Dict[str, int] = {}

        for strat in strategies:
            sid = strat.get("strategy_id", strat.get("name", "unknown"))
            name = strat.get("name", sid)

            passed, reasons = self._hard_filters(strat, config, pbo_scores.get(sid))
            for r in reasons:
                filter_stats[r] = filter_stats.get(r, 0) + 1

            score = self._composite_score(strat, config, pbo_scores.get(sid))

            ov_metrics: Dict[str, Any] = {}
            if sid in pbo_scores:
                ov_metrics["pbo"] = pbo_scores[sid]
            if self.detector and "returns" in strat:
                a = self.detector.analyze_strategy(strat["returns"], n_trials=len(strategies))
                ov_metrics["psr"] = a["psr"].psr
                ov_metrics["dsr"] = a["dsr"].deflated_sharpe
                ov_metrics["dsr_significant"] = a["dsr"].is_significant

            results.append(FilterResult(
                strategy_id=sid, name=name, passed=passed,
                rejection_reasons=reasons, composite_score=score,
                metrics=strat, overfitting_metrics=ov_metrics,
            ))

        # Step 3: Split passed / rejected
        passed_list = [r for r in results if r.passed]
        rejected_list = [r for r in results if not r.passed]

        # Step 4: Rank by composite score
        passed_list.sort(key=lambda r: r.composite_score, reverse=True)

        # Step 5: Select top N
        n_select = (
            max(1, int(len(passed_list) * config.top_pct / 100))
            if config.top_pct is not None else config.top_n
        )
        survivors = passed_list[:n_select]
        overflow = passed_list[n_select:]

        for i, s in enumerate(survivors):
            s.rank = i + 1
        for s in overflow:
            s.passed = False
            s.rejection_reasons.append("below_top_n_cutoff")
            rejected_list.append(s)

        # Step 6: Update lineage status
        if self.lineage:
            for s in survivors:
                self.lineage.update_status(s.strategy_id, "filtered")
            for s in rejected_list:
                self.lineage.update_status(s.strategy_id, "rejected")

        result = PipelineResult(
            total_input=len(strategies),
            total_passed_hard=len(survivors) + len(overflow),
            total_survivors=len(survivors),
            survivors=survivors, rejected=rejected_list,
            config=config, filter_stats=filter_stats,
        )
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # HARD FILTERS
    # ------------------------------------------------------------------
    def _hard_filters(
        self, s: Dict, cfg: FilterConfig, pbo: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        reasons = []
        sr = s.get("sharpe_ratio")
        if sr is not None and sr < cfg.min_sharpe:
            reasons.append(f"sharpe_below_{cfg.min_sharpe}")
        dd = s.get("max_drawdown_pct")
        if dd is not None and dd > cfg.max_drawdown:
            reasons.append(f"drawdown_above_{cfg.max_drawdown}%")
        tr = s.get("total_trades")
        if tr is not None and tr < cfg.min_trades:
            reasons.append(f"trades_below_{cfg.min_trades}")
        wr = s.get("win_rate")
        if wr is not None and cfg.min_win_rate > 0 and wr < cfg.min_win_rate:
            reasons.append(f"win_rate_below_{cfg.min_win_rate}")
        pf = s.get("profit_factor")
        if pf is not None and cfg.min_profit_factor > 0 and pf < cfg.min_profit_factor:
            reasons.append(f"profit_factor_below_{cfg.min_profit_factor}")
        ret = s.get("total_return_pct")
        if ret is not None and ret < cfg.min_total_return:
            reasons.append(f"return_below_{cfg.min_total_return}%")
        if pbo is not None and pbo > cfg.max_pbo:
            reasons.append(f"pbo_above_{cfg.max_pbo}")
        return (len(reasons) == 0, reasons)

    # ------------------------------------------------------------------
    # COMPOSITE SCORE
    # ------------------------------------------------------------------
    def _composite_score(
        self, s: Dict, cfg: FilterConfig, pbo: Optional[float] = None,
    ) -> float:
        score, weight = 0.0, 0.0

        def _add(val, w, lo, hi, invert=False):
            nonlocal score, weight
            if val is not None:
                n = np.clip((val - lo) / max(hi - lo, 1e-10), 0, 1)
                if invert:
                    n = 1.0 - n
                score += w * n
                weight += w

        _add(s.get("sharpe_ratio"),     cfg.weight_sharpe,         0, 3)
        _add(s.get("max_drawdown_pct"), cfg.weight_drawdown,       0, 50, invert=True)
        _add(s.get("profit_factor"),    cfg.weight_profit_factor,  0, 3)
        _add(s.get("win_rate"),         cfg.weight_win_rate,       0, 1)
        _add(s.get("total_trades"),     cfg.weight_trades,         0, 500)
        _add(pbo,                       cfg.weight_pbo,            0, 1,  invert=True)

        return score / weight if weight > 0 else 0.0

    # ------------------------------------------------------------------
    # PBO HELPER
    # ------------------------------------------------------------------
    def _compute_pbo_scores(self, rm: pd.DataFrame) -> Dict[str, float]:
        if self.detector is None or rm.shape[1] < 4:
            return {}
        try:
            n_part = min(16, max(4, rm.shape[0] // 50))
            if n_part % 2 != 0:
                n_part -= 1
            n_part = max(4, n_part)
            pbo = self.detector.compute_pbo(rm, n_partitions=n_part)
            return {col: pbo.probability for col in rm.columns}
        except Exception as e:
            print(f"  ⚠️  PBO failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # DATABASE LOADER
    # ------------------------------------------------------------------
    def load_from_database(self, db_path: str, table: str = "backtest_results") -> List[Dict]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    def save_results(self, result: PipelineResult, output_path: str):
        data = {
            "timestamp": result.timestamp,
            "total_input": result.total_input,
            "total_passed_hard": result.total_passed_hard,
            "total_survivors": result.total_survivors,
            "filter_stats": result.filter_stats,
            "survivors": [
                {
                    "strategy_id": s.strategy_id,
                    "name": s.name,
                    "rank": s.rank,
                    "composite_score": s.composite_score,
                    "sharpe_ratio": s.metrics.get("sharpe_ratio"),
                    "max_drawdown_pct": s.metrics.get("max_drawdown_pct"),
                    "total_trades": s.metrics.get("total_trades"),
                    "overfitting": s.overfitting_metrics,
                }
                for s in result.survivors
            ],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✓ Results saved to {output_path}")
