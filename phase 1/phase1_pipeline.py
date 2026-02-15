# ==============================================================================
# phase1_pipeline.py
# ==============================================================================
# Phase 1: Foundation Completion — Integration Pipeline
#
# Orchestrates all four modules end-to-end:
#   Module 1: LineageTracker     — register strategies, track genealogy
#   Module 2: OverfittingDetector — PBO + DSR overfitting analysis
#   Module 3: FilteringPipeline   — hard filters → rank → top N
#   Module 4: DiversificationFilter — correlation → greedy select
#
# Data flow:
#   strategies + returns
#       → register in lineage (M1)
#       → PBO analysis (M2)
#       → threshold filters + scoring (M3)
#       → correlation/diversification (M4)
#       → final pool ready for Phase 2 optimization
#
# Usage:
#     from phase1_pipeline import Phase1Pipeline
#
#     pipeline = Phase1Pipeline(db_path="data/lineage.db")
#     result = pipeline.run(
#         strategies=strategy_list,       # list of dicts with metrics
#         returns_dict=returns_by_id,     # {strategy_id: np.array}
#     )
#     print(result.summary())
#
# ==============================================================================

import numpy as np
import pandas as pd
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lineage_tracker import LineageTracker
from overfitting_detector import OverfittingDetector, PBOResult
from filtering_pipeline import FilteringPipeline, FilterConfig, PipelineResult
from diversification_filter import DiversificationFilter, DiversityConfig, DiversificationResult


# ==============================================================================
# PIPELINE RESULT
# ==============================================================================

@dataclass
class Phase1Result:
    """Complete output from Phase 1."""
    timestamp: str
    total_input: int
    total_registered: int
    total_passed_filters: int
    total_diversified: int
    pbo_result: Optional[PBOResult]
    filter_result: Optional[PipelineResult]
    diversity_result: Optional[DiversificationResult]
    final_strategies: List[Dict[str, Any]]

    def summary(self) -> str:
        pbo_str = f"{self.pbo_result.probability:.2%}" if self.pbo_result else "N/A"
        avg_corr = self.diversity_result.avg_pairwise_corr if self.diversity_result else 0.0
        eff_n = self.diversity_result.effective_n if self.diversity_result else 0
        lines = [
            f"\n{'='*70}",
            f"  PHASE 1 PIPELINE — FOUNDATION COMPLETION",
            f"{'='*70}",
            f"  Timestamp:         {self.timestamp}",
            f"  Input strategies:  {self.total_input}",
            f"  Registered:        {self.total_registered}",
            f"  Passed filters:    {self.total_passed_filters}",
            f"  Final diversified: {self.total_diversified}",
            f"{'—'*70}",
            f"  Avg PBO:           {pbo_str}",
            f"  Overfit rejected:  {self.filter_result.filter_stats.get('pbo_above_0.5', 0) if self.filter_result else 0}",
            f"  Avg pairwise corr: {avg_corr:.4f}",
            f"  Effective N:       {eff_n:.1f}",
            f"{'='*70}",
        ]
        if self.final_strategies:
            lines.append("  Top 5 final strategies:")
            for i, s in enumerate(self.final_strategies[:5]):
                sr = s.get("sharpe_ratio", s.get("composite_score", 0))
                lines.append(f"    {i+1}. {s.get('name','?')} "
                             f"(score={s.get('composite_score',0):.4f}, "
                             f"Sharpe={sr:.3f})")
        return "\n".join(lines)


# ==============================================================================
# PHASE 1 PIPELINE
# ==============================================================================

class Phase1Pipeline:
    """
    End-to-end Phase 1 orchestrator.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        enable_mlflow: bool = False,
        filter_config: Optional[FilterConfig] = None,
        diversity_config: Optional[DiversityConfig] = None,
    ):
        self.tracker = LineageTracker(db_path=db_path, enable_mlflow=enable_mlflow)
        self.detector = OverfittingDetector()
        self.filter_cfg = filter_config or FilterConfig()
        self.diversity_cfg = diversity_config or DiversityConfig()

        self.filterer = FilteringPipeline(
            lineage_tracker=self.tracker,
            overfitting_detector=self.detector,
        )
        self.diversifier = DiversificationFilter(lineage_tracker=self.tracker)

    def run(
        self,
        strategies: List[Dict[str, Any]],
        returns_dict: Optional[Dict[str, np.ndarray]] = None,
        trade_dates_dict: Optional[Dict[str, set]] = None,
    ) -> Phase1Result:
        """
        Run the full Phase 1 pipeline.

        Args:
            strategies: List of dicts. Required keys:
                strategy_id, name, sharpe_ratio, max_drawdown_pct,
                total_trades, total_return_pct.
                Optional: win_rate, profit_factor, origin, hypothesis.
            returns_dict: {strategy_id: np.array of daily returns}
            trade_dates_dict: {strategy_id: set of date strings}
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 1 PIPELINE — STARTING")
        print(f"{'='*70}")

        # ---- Step 1: Register in lineage ----
        print(f"\n📋 Step 1: Registering strategies in lineage tracker...")
        registered = 0
        for s in strategies:
            sid = s.get("strategy_id", s.get("name"))
            self.tracker.register_strategy(
                name=s.get("name", sid),
                origin=s.get("origin", "discovered"),
                strategy_id=sid,
                hypothesis=s.get("hypothesis"),
                code_hash=s.get("code_hash"),
            )
            # Log metrics if present
            metric_keys = ["sharpe_ratio", "max_drawdown_pct", "total_return_pct",
                           "total_trades", "win_rate", "profit_factor"]
            metrics = {k: s[k] for k in metric_keys if k in s and s[k] is not None}
            if metrics:
                self.tracker.log_backtest(sid, metrics)
            registered += 1
        print(f"  ✓ Registered {registered}/{len(strategies)} strategies")

        # ---- Step 2: PBO analysis ----
        pbo_result = None
        returns_matrix = None
        if returns_dict:
            print(f"\n🔍 Step 2: Running overfitting analysis...")
            # Build returns DataFrame
            shared_ids = [s.get("strategy_id", s.get("name"))
                          for s in strategies if s.get("strategy_id", s.get("name")) in returns_dict]
            if len(shared_ids) >= 4:
                min_len = min(len(returns_dict[sid]) for sid in shared_ids)
                returns_matrix = pd.DataFrame(
                    {sid: returns_dict[sid][:min_len] for sid in shared_ids}
                )
                n_part = min(16, max(4, min_len // 50))
                if n_part % 2 != 0:
                    n_part -= 1
                n_part = max(4, n_part)
                pbo_result = self.detector.compute_pbo(returns_matrix, n_partitions=n_part)
                print(f"  PBO = {pbo_result.probability:.2%} "
                      f"({'✅ OK' if not pbo_result.is_overfit else '⚠️  OVERFIT'})")
        else:
            print(f"\n🔍 Step 2: Running overfitting analysis...")
            print(f"  (No returns provided — skipping PBO)")

        # ---- Step 3: Filtering pipeline ----
        print(f"\n🔬 Step 3: Running filtering pipeline...")
        filter_result = self.filterer.run(
            strategies=strategies,
            config=self.filter_cfg,
            returns_matrix=returns_matrix,
        )

        # ---- Step 4: Diversification filter ----
        print(f"\n🎯 Step 4: Running diversification filter...")
        survivor_dicts = []
        for s in filter_result.survivors:
            d = dict(s.metrics)
            d["strategy_id"] = s.strategy_id
            d["name"] = s.name
            d["composite_score"] = s.composite_score
            survivor_dicts.append(d)

        survivor_returns = {}
        if returns_dict:
            for d in survivor_dicts:
                sid = d["strategy_id"]
                if sid in returns_dict:
                    survivor_returns[sid] = returns_dict[sid]

        diversity_result = self.diversifier.run(
            strategies=survivor_dicts,
            returns_dict=survivor_returns if survivor_returns else None,
            trade_dates_dict=trade_dates_dict,
            config=self.diversity_cfg,
        )

        # ---- Build final output ----
        final = diversity_result.selected

        # Save results
        output_dir = Path(self.tracker.db_path).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        phase1_data = {
            "timestamp": datetime.now().isoformat(),
            "total_input": len(strategies),
            "registered": registered,
            "passed_filters": filter_result.total_survivors,
            "final_diversified": len(final),
            "pbo": pbo_result.probability if pbo_result else None,
            "avg_corr": diversity_result.avg_pairwise_corr,
            "effective_n": diversity_result.effective_n,
            "strategies": [
                {"strategy_id": s.get("strategy_id"), "name": s.get("name"),
                 "composite_score": s.get("composite_score"),
                 "sharpe_ratio": s.get("sharpe_ratio")}
                for s in final
            ],
        }
        with open(output_dir / "phase1_results.json", "w") as f:
            json.dump(phase1_data, f, indent=2, default=str)
        print(f"✓ Results saved to {output_dir / 'phase1_results.json'}")

        self.filterer.save_results(filter_result, str(output_dir / "filter_results.json"))

        result = Phase1Result(
            timestamp=datetime.now().isoformat(),
            total_input=len(strategies),
            total_registered=registered,
            total_passed_filters=filter_result.total_survivors,
            total_diversified=len(final),
            pbo_result=pbo_result,
            filter_result=filter_result,
            diversity_result=diversity_result,
            final_strategies=final,
        )
        print(result.summary())
        return result
