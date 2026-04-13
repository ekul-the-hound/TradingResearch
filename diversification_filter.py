# ==============================================================================
# diversification_filter.py
# ==============================================================================
# Module 4 of 4 -- Phase 1: Foundation Completion
#
# Correlation & Diversification Filter
#
# Removes redundant strategies from the survivor pool using:
#   1. Return correlation matrix (Pearson on daily returns)
#   2. Trade overlap analysis (Jaccard on trade dates)
#   3. Greedy selection: pick highest-score strategy, add next if
#      correlation with all selected < threshold, repeat.
#
# No external GitHub repos -- numpy/scipy correlation.
#
# Consumed by:
#   - phase1_pipeline.py (final stage of Phase 1)
#   - Phase 2 NSGA-II (initial population = diversified survivors)
#
# Usage:
#     from diversification_filter import DiversificationFilter, DiversityConfig
#
#     filt = DiversificationFilter()
#     result = filt.run(
#         strategies=[{"strategy_id": "s1", "name": "S1", "composite_score": 0.8}],
#         returns_dict={"s1": np.array([...])},
#         config=DiversityConfig(max_correlation=0.5)
#     )
#     print(result.summary())
#
# ==============================================================================

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class DiversityConfig:
    """Configuration for the diversification filter."""
    max_correlation: float = 0.50    # Max pairwise |ρ| between any two selected
    max_strategies: int = 50         # Hard cap on output size
    min_strategies: int = 5          # Keep at least this many (relaxes ρ if needed)
    trade_overlap_weight: float = 0.3  # Blend: (1-w)*return_corr + w*trade_overlap
    cluster_threshold: float = 0.7   # Distance threshold for clustering


# ==============================================================================
# RESULT DATACLASS
# ==============================================================================

@dataclass
class DiversificationResult:
    """Output of the diversification filter."""
    selected: List[Dict[str, Any]]        # Strategies that passed
    removed: List[Dict[str, Any]]         # Strategies removed as redundant
    correlation_matrix: Optional[pd.DataFrame]
    avg_pairwise_corr: float
    max_pairwise_corr: float
    effective_n: float                    # 1/sum(w_i^2) diversification measure
    n_clusters: int
    cluster_labels: Optional[Dict[str, int]]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  DIVERSIFICATION FILTER RESULTS",
            f"{'='*60}",
            f"  Input strategies:  {len(self.selected) + len(self.removed)}",
            f"  Selected:          {len(self.selected)}",
            f"  Removed redundant: {len(self.removed)}",
            f"  Avg pairwise corr: {self.avg_pairwise_corr:.4f}",
            f"  Max pairwise corr: {self.max_pairwise_corr:.4f}",
            f"  Effective N:       {self.effective_n:.1f}",
        ]
        if self.selected:
            lines.append(f"\n  Selected strategies:")
            for i, s in enumerate(self.selected[:10]):
                lines.append(f"    {i+1}. {s['name']} (score={s.get('composite_score', 0):.4f})")
            if len(self.selected) > 10:
                lines.append(f"    ... and {len(self.selected)-10} more")
        return "\n".join(lines)


# ==============================================================================
# DIVERSIFICATION FILTER
# ==============================================================================

class DiversificationFilter:
    """
    Greedy correlation-based strategy selector.

    Algorithm:
      1. Sort strategies by composite_score descending.
      2. Select the top strategy.
      3. For each remaining strategy (by score):
         a. Compute combined similarity to ALL already-selected.
         b. If max similarity < threshold -> select it.
         c. Else -> remove as redundant.
      4. If selected < min_strategies, relax threshold and retry.
    """

    def __init__(self, lineage_tracker: Optional[Any] = None):
        self.lineage = lineage_tracker

    def run(
        self,
        strategies: List[Dict[str, Any]],
        returns_dict: Optional[Dict[str, np.ndarray]] = None,
        trade_dates_dict: Optional[Dict[str, Set[str]]] = None,
        config: Optional[DiversityConfig] = None,
    ) -> DiversificationResult:
        """
        Run the diversification filter.

        Args:
            strategies: List of dicts with 'strategy_id', 'name',
                       'composite_score'. Ordered by score or will be sorted.
            returns_dict: {strategy_id: np.array of daily returns}
            trade_dates_dict: {strategy_id: set of date strings} for overlap.
            config: DiversityConfig thresholds.
        """
        if config is None:
            config = DiversityConfig()

        n = len(strategies)
        print(f"\n[TARGET] Diversification Filter: {n} strategies -> max {config.max_strategies}")

        if n == 0:
            return self._empty_result(config)

        # Sort by composite score descending
        strategies = sorted(strategies, key=lambda s: s.get("composite_score", 0), reverse=True)
        sids = [s.get("strategy_id", s.get("name")) for s in strategies]

        # Build similarity matrix
        sim_matrix = self._build_similarity_matrix(
            sids, returns_dict, trade_dates_dict, config.trade_overlap_weight,
        )

        # Greedy selection
        selected_idx, removed_idx = self._greedy_select(
            strategies, sim_matrix, config.max_correlation,
            config.max_strategies, config.min_strategies,
        )

        selected = [strategies[i] for i in selected_idx]
        removed = [strategies[i] for i in removed_idx]

        # Correlation matrix for selected
        corr_df, avg_corr, max_corr = self._compute_selected_stats(
            selected_idx, sids, sim_matrix,
        )

        # Effective N (inverse HHI on equal weights)
        eff_n = len(selected) if len(selected) > 0 else 0
        if eff_n > 1 and corr_df is not None:
            try:
                vals = corr_df.values
                np.fill_diagonal(vals, 0)
                mean_abs = np.mean(np.abs(vals))
                eff_n = len(selected) * (1 - mean_abs)
            except Exception:
                pass

        # Clustering
        n_clusters = 0
        cluster_labels = None
        if len(selected) >= 3 and sim_matrix is not None:
            cluster_labels, n_clusters = self._cluster_strategies(
                selected_idx, sim_matrix, config.cluster_threshold,
                [s.get("name", s.get("strategy_id")) for s in selected],
            )

        # Update lineage
        if self.lineage:
            for s in selected:
                self.lineage.update_status(
                    s.get("strategy_id", s.get("name")), "diversified"
                )

        result = DiversificationResult(
            selected=selected, removed=removed,
            correlation_matrix=corr_df,
            avg_pairwise_corr=avg_corr, max_pairwise_corr=max_corr,
            effective_n=eff_n, n_clusters=n_clusters,
            cluster_labels=cluster_labels,
        )
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # SIMILARITY MATRIX
    # ------------------------------------------------------------------
    def _build_similarity_matrix(
        self,
        sids: List[str],
        returns_dict: Optional[Dict[str, np.ndarray]],
        trade_dates_dict: Optional[Dict[str, Set[str]]],
        overlap_weight: float,
    ) -> Optional[np.ndarray]:
        n = len(sids)
        if n < 2:
            return None

        # Return correlation
        ret_corr = np.zeros((n, n))
        if returns_dict:
            for i in range(n):
                for j in range(i + 1, n):
                    ri = returns_dict.get(sids[i])
                    rj = returns_dict.get(sids[j])
                    if ri is not None and rj is not None:
                        min_len = min(len(ri), len(rj))
                        if min_len > 3:
                            c = np.corrcoef(ri[:min_len], rj[:min_len])[0, 1]
                            if np.isfinite(c):
                                ret_corr[i, j] = ret_corr[j, i] = abs(c)

        # Trade overlap (Jaccard)
        overlap = np.zeros((n, n))
        if trade_dates_dict:
            for i in range(n):
                for j in range(i + 1, n):
                    di = trade_dates_dict.get(sids[i], set())
                    dj = trade_dates_dict.get(sids[j], set())
                    if di and dj:
                        union = len(di | dj)
                        if union > 0:
                            jac = len(di & dj) / union
                            overlap[i, j] = overlap[j, i] = jac

        # Blend
        w = overlap_weight if trade_dates_dict else 0.0
        sim = (1 - w) * ret_corr + w * overlap
        return sim

    # ------------------------------------------------------------------
    # GREEDY SELECTION
    # ------------------------------------------------------------------
    def _greedy_select(
        self,
        strategies: List[Dict],
        sim: Optional[np.ndarray],
        threshold: float,
        max_strats: int,
        min_strats: int,
    ) -> tuple:
        n = len(strategies)
        if sim is None or n <= 1:
            sel = list(range(min(n, max_strats)))
            rem = list(range(len(sel), n))
            return sel, rem

        selected: List[int] = [0]  # Always pick the best-scored
        removed: List[int] = []

        for i in range(1, n):
            if len(selected) >= max_strats:
                removed.append(i)
                continue

            # Check max similarity against all currently selected
            max_sim = max(sim[i, j] for j in selected)

            if max_sim < threshold:
                selected.append(i)
            else:
                removed.append(i)

        # If we have fewer than min_strats, relax threshold
        if len(selected) < min_strats and removed:
            remaining = sorted(removed, key=lambda i: strategies[i].get("composite_score", 0), reverse=True)
            while len(selected) < min_strats and remaining:
                selected.append(remaining.pop(0))
            removed = remaining

        return selected, removed

    # ------------------------------------------------------------------
    # STATS FOR SELECTED
    # ------------------------------------------------------------------
    def _compute_selected_stats(
        self,
        selected_idx: List[int],
        sids: List[str],
        sim: Optional[np.ndarray],
    ) -> tuple:
        if sim is None or len(selected_idx) < 2:
            return None, 0.0, 0.0

        names = [sids[i] for i in selected_idx]
        sub = sim[np.ix_(selected_idx, selected_idx)]
        df = pd.DataFrame(sub, index=names, columns=names)

        mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
        upper = sub[mask]
        avg_c = float(np.mean(upper)) if len(upper) > 0 else 0.0
        max_c = float(np.max(upper)) if len(upper) > 0 else 0.0
        return df, avg_c, max_c

    # ------------------------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------------------------
    def _cluster_strategies(
        self,
        selected_idx: List[int],
        sim: np.ndarray,
        threshold: float,
        names: List[str],
    ) -> tuple:
        try:
            sub = sim[np.ix_(selected_idx, selected_idx)]
            dist = 1.0 - sub
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, None)
            dist = (dist + dist.T) / 2

            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=threshold, criterion="distance")
            n_clusters = len(set(labels))
            label_dict = {names[i]: int(labels[i]) for i in range(len(names))}
            return label_dict, n_clusters
        except Exception:
            return None, 0

    # ------------------------------------------------------------------
    # TRADE OVERLAP (UTILITY)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_trade_overlap(dates_a: Set[str], dates_b: Set[str]) -> float:
        """Jaccard similarity between two sets of trade dates."""
        if not dates_a or not dates_b:
            return 0.0
        union = len(dates_a | dates_b)
        return len(dates_a & dates_b) / union if union > 0 else 0.0

    def _empty_result(self, config):
        return DiversificationResult(
            selected=[], removed=[], correlation_matrix=None,
            avg_pairwise_corr=0.0, max_pairwise_corr=0.0,
            effective_n=0, n_clusters=0, cluster_labels=None,
        )
