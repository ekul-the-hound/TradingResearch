# ==============================================================================
# lineage_analytics.py
# ==============================================================================
# Phase 6, Module 2 (Week 24): Lineage Analytics & Bias Learning
#
# Answers the meta-questions: What works? What doesn't? What's decaying?
# Analyzes the entire strategy genealogy to learn biases that improve
# future search and mutation decisions.
#
# Analytics:
#   1. Mutation effectiveness: which mutation types improve Sharpe?
#   2. Hypothesis decay: which research ideas are going stale?
#   3. Genealogy depth: how many generations produce improvement?
#   4. Feature attribution: what strategy features predict success?
#   5. Regime bias: which strategies work in which regimes?
#   6. Cost-effectiveness: ROI per mutation type (compute vs improvement)
#
# Consumed by:
#   - learning_loop.py (prune decisions, mutation weighting)
#   - optimization_pipeline.py (informed search space)
#   - dashboard (analytics pages)
#
# Usage:
#     from lineage_analytics import LineageAnalyzer
#     analyzer = LineageAnalyzer()
#     analyzer.add_strategy(strategy_record)
#     report = analyzer.generate_report()
# ==============================================================================

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class StrategyLineage:
    """Lineage record for one strategy."""
    strategy_id: str
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    hypothesis_id: Optional[str] = None
    generation: int = 0
    created_at: str = ""

    # Performance
    backtest_sharpe: float = 0.0
    live_sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0

    # Cost
    compute_seconds: float = 0.0
    api_cost_usd: float = 0.0

    # Regime performance
    regime_sharpes: Dict[str, float] = field(default_factory=dict)

    # Status
    is_active: bool = True
    final_state: str = "research"  # research, paper, live, retired, rejected


@dataclass
class MutationReport:
    """Analysis of one mutation type."""
    mutation_type: str
    total_count: int
    success_rate: float         # % that improved over parent
    avg_improvement: float      # Average Sharpe delta
    median_improvement: float
    best_improvement: float
    worst_degradation: float
    avg_compute_cost: float
    roi: float                  # Improvement per dollar of compute


@dataclass
class HypothesisReport:
    """Analysis of one research hypothesis."""
    hypothesis_id: str
    total_strategies: int
    active_strategies: int
    avg_sharpe: float
    sharpe_trend: float         # Slope: positive = improving, negative = decaying
    days_since_creation: int
    is_stale: bool
    best_strategy_id: Optional[str]
    best_sharpe: float


@dataclass
class GenerationReport:
    """Performance by generation depth."""
    generation: int
    n_strategies: int
    avg_sharpe: float
    avg_improvement: float      # vs previous generation
    survival_rate: float        # % that made it to paper/live


@dataclass
class AnalyticsReport:
    """Complete analytics output."""
    timestamp: str
    total_strategies: int
    active_strategies: int
    total_mutations: int

    # Per-mutation-type analysis
    mutation_reports: List[MutationReport]
    best_mutation: Optional[str]
    worst_mutation: Optional[str]

    # Hypothesis health
    hypothesis_reports: List[HypothesisReport]
    stale_hypotheses: int
    healthy_hypotheses: int

    # Generation analysis
    generation_reports: List[GenerationReport]
    optimal_depth: int          # Generation with best avg Sharpe

    # Recommended weights for future mutations
    mutation_weights: Dict[str, float]

    # Regime insights
    regime_best: Dict[str, str]  # regime -> best mutation type

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  LINEAGE ANALYTICS REPORT",
            f"{'='*60}",
            f"  Strategies:    {self.total_strategies} total, {self.active_strategies} active",
            f"  Mutations:     {self.total_mutations}",
            f"  Best mutation: {self.best_mutation or 'N/A'}",
            f"  Worst mutation:{self.worst_mutation or 'N/A'}",
            f"  Hypotheses:    {self.healthy_hypotheses} healthy, {self.stale_hypotheses} stale",
            f"  Optimal depth: Generation {self.optimal_depth}",
            f"\n  Mutation Weights (recommended):",
        ]
        for mt, w in sorted(self.mutation_weights.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {mt:30s} -> {w:.3f}")
        return "\n".join(lines)


# ==============================================================================
# LINEAGE ANALYZER
# ==============================================================================

class LineageAnalyzer:
    """Genealogy analytics for strategy improvement."""

    def __init__(self):
        self._strategies: Dict[str, StrategyLineage] = {}

    # ------------------------------------------------------------------
    # DATA INGESTION
    # ------------------------------------------------------------------
    def add_strategy(self, record: StrategyLineage):
        self._strategies[record.strategy_id] = record

    def add_from_dict(self, d: Dict[str, Any]):
        self._strategies[d["strategy_id"]] = StrategyLineage(**{
            k: v for k, v in d.items()
            if k in StrategyLineage.__dataclass_fields__
        })

    def add_batch(self, records: List[StrategyLineage]):
        for r in records:
            self._strategies[r.strategy_id] = r

    # ------------------------------------------------------------------
    # MAIN REPORT
    # ------------------------------------------------------------------
    def generate_report(self) -> AnalyticsReport:
        ts = datetime.now().isoformat()
        strats = list(self._strategies.values())
        active = [s for s in strats if s.is_active]

        mutation_reports = self._analyze_mutations(strats)
        hypothesis_reports = self._analyze_hypotheses(strats)
        generation_reports = self._analyze_generations(strats)
        mutation_weights = self._compute_mutation_weights(mutation_reports)
        regime_best = self._analyze_regime_bias(strats)

        best_mut = max(mutation_reports, key=lambda r: r.avg_improvement).mutation_type if mutation_reports else None
        worst_mut = min(mutation_reports, key=lambda r: r.avg_improvement).mutation_type if mutation_reports else None
        optimal_gen = max(generation_reports, key=lambda r: r.avg_sharpe).generation if generation_reports else 0

        return AnalyticsReport(
            timestamp=ts,
            total_strategies=len(strats),
            active_strategies=len(active),
            total_mutations=sum(1 for s in strats if s.mutation_type),
            mutation_reports=mutation_reports,
            best_mutation=best_mut,
            worst_mutation=worst_mut,
            hypothesis_reports=hypothesis_reports,
            stale_hypotheses=sum(1 for h in hypothesis_reports if h.is_stale),
            healthy_hypotheses=sum(1 for h in hypothesis_reports if not h.is_stale),
            generation_reports=generation_reports,
            optimal_depth=optimal_gen,
            mutation_weights=mutation_weights,
            regime_best=regime_best,
        )

    # ------------------------------------------------------------------
    # MUTATION ANALYSIS
    # ------------------------------------------------------------------
    def _analyze_mutations(self, strats: List[StrategyLineage]) -> List[MutationReport]:
        by_type: Dict[str, List[Tuple[StrategyLineage, StrategyLineage]]] = defaultdict(list)

        for s in strats:
            if s.mutation_type and s.parent_id and s.parent_id in self._strategies:
                parent = self._strategies[s.parent_id]
                by_type[s.mutation_type].append((s, parent))

        reports = []
        for mt, pairs in by_type.items():
            improvements = [child.live_sharpe - parent.live_sharpe for child, parent in pairs]
            arr = np.array(improvements)
            costs = [child.compute_seconds + child.api_cost_usd * 100 for child, _ in pairs]
            avg_cost = np.mean(costs) if costs else 0
            avg_imp = float(np.mean(arr))
            roi = avg_imp / max(avg_cost, 0.01)

            reports.append(MutationReport(
                mutation_type=mt,
                total_count=len(pairs),
                success_rate=float(np.mean(arr > 0)),
                avg_improvement=avg_imp,
                median_improvement=float(np.median(arr)),
                best_improvement=float(np.max(arr)),
                worst_degradation=float(np.min(arr)),
                avg_compute_cost=float(avg_cost),
                roi=roi,
            ))

        return sorted(reports, key=lambda r: -r.avg_improvement)

    # ------------------------------------------------------------------
    # HYPOTHESIS ANALYSIS
    # ------------------------------------------------------------------
    def _analyze_hypotheses(self, strats: List[StrategyLineage]) -> List[HypothesisReport]:
        by_hyp: Dict[str, List[StrategyLineage]] = defaultdict(list)
        for s in strats:
            if s.hypothesis_id:
                by_hyp[s.hypothesis_id].append(s)

        reports = []
        for hid, group in by_hyp.items():
            sharpes = [s.live_sharpe for s in group if s.live_sharpe != 0]
            active = [s for s in group if s.is_active]
            avg_sharpe = float(np.mean(sharpes)) if sharpes else 0

            # Trend: simple linear regression on sharpes over time
            if len(sharpes) >= 3:
                x = np.arange(len(sharpes), dtype=float)
                slope = float(np.polyfit(x, sharpes, 1)[0])
            else:
                slope = 0.0

            # Days since first strategy
            dates = [s.created_at for s in group if s.created_at]
            days = 0
            if dates:
                try:
                    first = min(dates)
                    days = (datetime.now() - datetime.fromisoformat(first)).days
                except (ValueError, TypeError):
                    days = 0

            best = max(group, key=lambda s: s.live_sharpe) if group else None
            is_stale = slope < -0.01 and days > 30

            reports.append(HypothesisReport(
                hypothesis_id=hid,
                total_strategies=len(group),
                active_strategies=len(active),
                avg_sharpe=avg_sharpe,
                sharpe_trend=slope,
                days_since_creation=days,
                is_stale=is_stale,
                best_strategy_id=best.strategy_id if best else None,
                best_sharpe=best.live_sharpe if best else 0,
            ))

        return sorted(reports, key=lambda r: -r.avg_sharpe)

    # ------------------------------------------------------------------
    # GENERATION ANALYSIS
    # ------------------------------------------------------------------
    def _analyze_generations(self, strats: List[StrategyLineage]) -> List[GenerationReport]:
        by_gen: Dict[int, List[StrategyLineage]] = defaultdict(list)
        for s in strats:
            by_gen[s.generation].append(s)

        reports = []
        prev_sharpe = 0.0
        for gen in sorted(by_gen.keys()):
            group = by_gen[gen]
            sharpes = [s.live_sharpe for s in group]
            avg = float(np.mean(sharpes)) if sharpes else 0
            survived = sum(1 for s in group if s.final_state in ("paper", "live"))
            survival = survived / max(len(group), 1)

            reports.append(GenerationReport(
                generation=gen,
                n_strategies=len(group),
                avg_sharpe=avg,
                avg_improvement=avg - prev_sharpe,
                survival_rate=survival,
            ))
            prev_sharpe = avg

        return reports

    # ------------------------------------------------------------------
    # MUTATION WEIGHTS
    # ------------------------------------------------------------------
    def _compute_mutation_weights(self, reports: List[MutationReport]) -> Dict[str, float]:
        """
        Compute recommended sampling weights for each mutation type.
        Better mutations get higher weight.
        Uses softmax on success_rate × avg_improvement.
        """
        if not reports:
            return {}

        scores = {}
        for r in reports:
            score = r.success_rate * max(r.avg_improvement + 0.1, 0.01)
            scores[r.mutation_type] = max(score, 0.01)

        # Softmax
        total = sum(scores.values())
        return {mt: s / total for mt, s in scores.items()}

    # ------------------------------------------------------------------
    # REGIME BIAS
    # ------------------------------------------------------------------
    def _analyze_regime_bias(self, strats: List[StrategyLineage]) -> Dict[str, str]:
        """Find which mutation types work best in each regime."""
        regime_mut: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for s in strats:
            if s.mutation_type and s.regime_sharpes:
                for regime, sharpe in s.regime_sharpes.items():
                    regime_mut[regime][s.mutation_type].append(sharpe)

        best = {}
        for regime, by_mut in regime_mut.items():
            best_mt = max(by_mut, key=lambda mt: np.mean(by_mut[mt]))
            best[regime] = best_mt

        return best

    # ------------------------------------------------------------------
    # GENEALOGY TREE
    # ------------------------------------------------------------------
    def get_family_tree(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get the full ancestor chain for a strategy."""
        chain = []
        current = strategy_id
        visited = set()
        while current and current in self._strategies and current not in visited:
            visited.add(current)
            s = self._strategies[current]
            chain.append({
                "strategy_id": s.strategy_id,
                "generation": s.generation,
                "mutation_type": s.mutation_type,
                "backtest_sharpe": s.backtest_sharpe,
                "live_sharpe": s.live_sharpe,
            })
            current = s.parent_id
        return chain

    def get_descendants(self, strategy_id: str) -> List[str]:
        """Get all children/grandchildren of a strategy."""
        children = [
            s.strategy_id for s in self._strategies.values()
            if s.parent_id == strategy_id
        ]
        all_desc = list(children)
        for c in children:
            all_desc.extend(self.get_descendants(c))
        return all_desc

    def get_top_lineages(self, n: int = 5) -> List[Dict[str, Any]]:
        """Find the most successful lineage chains."""
        # Find strategies with highest live Sharpe and trace back
        sorted_strats = sorted(
            self._strategies.values(),
            key=lambda s: s.live_sharpe, reverse=True,
        )
        seen_roots = set()
        lineages = []
        for s in sorted_strats[:n * 3]:
            tree = self.get_family_tree(s.strategy_id)
            root = tree[-1]["strategy_id"] if tree else s.strategy_id
            if root not in seen_roots:
                seen_roots.add(root)
                lineages.append({
                    "root": root,
                    "best": s.strategy_id,
                    "best_sharpe": s.live_sharpe,
                    "depth": len(tree),
                    "chain": tree,
                })
            if len(lineages) >= n:
                break
        return lineages
