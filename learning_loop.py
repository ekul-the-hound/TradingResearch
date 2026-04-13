# ==============================================================================
# learning_loop.py
# ==============================================================================
# Phase 6, Module 1 (Week 23): Self-Improving Learning Loop
#
# The brain that makes TradingLab continuously improve. Orchestrates
# retraining, surrogate refresh, acquisition adaptation, and hypothesis
# pruning based on live performance feedback.
#
# Triggers:
#   1. Drift detected -> retrain affected strategies
#   2. New data threshold -> refresh surrogate model
#   3. Scheduled interval -> periodic maintenance
#   4. Performance degradation -> parameter adjustment
#   5. Manual request -> on-demand cycle
#
# Each cycle:
#   a. Collect performance data from live/shadow traders
#   b. Identify strategies needing attention (drift, degradation)
#   c. Refresh surrogate model with new backtest results
#   d. Update acquisition function weights (explore/exploit)
#   e. Trigger re-optimization for degraded strategies
#   f. Prune dead hypotheses
#   g. Log everything for lineage_analytics.py
#
# Consumed by:
#   - Main scheduler / cron
#   - Dashboard (manual trigger)
#
# Usage:
#     from learning_loop import LearningLoop, LoopConfig
#     loop = LearningLoop(config=LoopConfig())
#     loop.register_strategy("S_001", backtest_returns, live_returns)
#     result = loop.run_cycle()
# ==============================================================================

import numpy as np
import json
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class TriggerType(Enum):
    DRIFT = "drift"
    DATA_THRESHOLD = "data_threshold"
    SCHEDULED = "scheduled"
    DEGRADATION = "degradation"
    MANUAL = "manual"


class ActionType(Enum):
    RETRAIN = "retrain"
    REFRESH_SURROGATE = "refresh_surrogate"
    ADJUST_ACQUISITION = "adjust_acquisition"
    PRUNE_HYPOTHESIS = "prune_hypothesis"
    DEMOTE_STRATEGY = "demote_strategy"
    FLAG_FOR_REVIEW = "flag_for_review"
    SKIP = "skip"


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class LoopConfig:
    """Learning loop thresholds and intervals."""
    # Data thresholds
    min_new_backtests_for_refresh: int = 50
    min_new_trades_for_analysis: int = 20

    # Scheduled
    retrain_interval_days: int = 7
    surrogate_refresh_interval_days: int = 3
    prune_interval_days: int = 14

    # Degradation
    sharpe_degradation_trigger_pct: float = 30.0
    drawdown_trigger_pct: float = 15.0

    # Acquisition adaptation
    acquisition_learning_rate: float = 0.1
    min_exploitation_ratio: float = 0.3
    max_exploitation_ratio: float = 0.9

    # Pruning
    hypothesis_stale_days: int = 90
    hypothesis_min_improvement_rate: float = 0.1  # 10% of children beat parent

    # Persistence
    state_dir: str = "data/learning_loop"
    verbose: bool = True


# ==============================================================================
# RESULTS
# ==============================================================================

@dataclass
class LoopAction:
    """One action taken during a cycle."""
    action: ActionType
    trigger: TriggerType
    strategy_id: Optional[str]
    details: str
    timestamp: str


@dataclass
class CycleResult:
    """Output from one learning cycle."""
    cycle_id: int
    timestamp: str
    triggers_fired: List[TriggerType]
    actions_taken: List[LoopAction]
    strategies_retrained: int
    strategies_demoted: int
    strategies_flagged: int
    surrogate_refreshed: bool
    hypotheses_pruned: int
    acquisition_kappa: float
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  LEARNING LOOP -- Cycle #{self.cycle_id}",
            f"{'='*60}",
            f"  Triggers:    {[t.value for t in self.triggers_fired]}",
            f"  Actions:     {len(self.actions_taken)}",
            f"  Retrained:   {self.strategies_retrained}",
            f"  Demoted:     {self.strategies_demoted}",
            f"  Flagged:     {self.strategies_flagged}",
            f"  Surrogate:   {'refreshed' if self.surrogate_refreshed else 'unchanged'}",
            f"  Pruned:      {self.hypotheses_pruned} hypotheses",
            f"  κ (explore): {self.acquisition_kappa:.3f}",
            f"  Elapsed:     {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


# ==============================================================================
# STRATEGY STATE
# ==============================================================================

@dataclass
class StrategyState:
    """Tracked state for one strategy in the learning loop."""
    strategy_id: str
    backtest_sharpe: float = 0.0
    live_sharpe: float = 0.0
    backtest_returns: Optional[np.ndarray] = None
    live_returns: Optional[np.ndarray] = None
    last_retrain: Optional[str] = None
    drift_detected: bool = False
    degradation_pct: float = 0.0
    hypothesis_id: Optional[str] = None
    mutation_type: Optional[str] = None
    parent_id: Optional[str] = None
    n_new_trades: int = 0


# ==============================================================================
# LEARNING LOOP
# ==============================================================================

class LearningLoop:
    """
    Self-improving orchestrator that closes the feedback loop.
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        retrain_fn: Optional[Callable] = None,
        surrogate_refresh_fn: Optional[Callable] = None,
        demote_fn: Optional[Callable] = None,
    ):
        self.config = config or LoopConfig()
        self.retrain_fn = retrain_fn
        self.surrogate_refresh_fn = surrogate_refresh_fn
        self.demote_fn = demote_fn

        self._strategies: Dict[str, StrategyState] = {}
        self._cycle_count = 0
        self._new_backtests = 0
        self._last_surrogate_refresh: Optional[str] = None
        self._last_prune: Optional[str] = None
        self._kappa = 2.0  # Current exploration parameter
        self._history: List[CycleResult] = []

        # Acquisition adaptation tracking
        self._acquisition_scores: Dict[str, List[float]] = {
            "ei": [], "ucb": [], "pi": [], "thompson": [],
        }

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------
    def register_strategy(
        self,
        strategy_id: str,
        backtest_returns: Optional[np.ndarray] = None,
        live_returns: Optional[np.ndarray] = None,
        backtest_sharpe: float = 0.0,
        hypothesis_id: Optional[str] = None,
        mutation_type: Optional[str] = None,
        parent_id: Optional[str] = None,
    ):
        bt_sharpe = backtest_sharpe
        if bt_sharpe == 0 and backtest_returns is not None and len(backtest_returns) > 10:
            r = backtest_returns
            bt_sharpe = float(np.mean(r) / max(np.std(r, ddof=1), 1e-10) * np.sqrt(252))

        self._strategies[strategy_id] = StrategyState(
            strategy_id=strategy_id,
            backtest_sharpe=bt_sharpe,
            backtest_returns=backtest_returns,
            live_returns=live_returns,
            hypothesis_id=hypothesis_id,
            mutation_type=mutation_type,
            parent_id=parent_id,
        )

    def update_live_data(
        self,
        strategy_id: str,
        live_returns: np.ndarray,
        drift_detected: bool = False,
        n_new_trades: int = 0,
    ):
        """Push new live data for a strategy."""
        if strategy_id not in self._strategies:
            return
        s = self._strategies[strategy_id]
        s.live_returns = live_returns
        s.drift_detected = drift_detected
        s.n_new_trades = n_new_trades
        if len(live_returns) > 10:
            s.live_sharpe = float(
                np.mean(live_returns) / max(np.std(live_returns, ddof=1), 1e-10) * np.sqrt(252)
            )
        if s.backtest_sharpe > 0:
            s.degradation_pct = (1 - s.live_sharpe / s.backtest_sharpe) * 100

    def add_backtest_results(self, n: int = 1):
        """Record that N new backtests have completed."""
        self._new_backtests += n

    # ------------------------------------------------------------------
    # MAIN CYCLE
    # ------------------------------------------------------------------
    def run_cycle(self, trigger: TriggerType = TriggerType.SCHEDULED) -> CycleResult:
        """Execute one learning cycle."""
        start = time.time()
        self._cycle_count += 1
        ts = datetime.now().isoformat()
        cfg = self.config
        _log = print if cfg.verbose else lambda *a, **k: None

        _log(f"\n  [CYCLE] Learning Loop -- Cycle #{self._cycle_count} ({trigger.value})")

        triggers = [trigger]
        actions: List[LoopAction] = []
        retrained = 0
        demoted = 0
        flagged = 0
        surrogate_refreshed = False
        pruned = 0

        # 1. Check drift-triggered retraining
        for sid, s in self._strategies.items():
            if s.drift_detected:
                if trigger not in triggers:
                    triggers.append(TriggerType.DRIFT)
                action = self._handle_drift(s, ts)
                actions.append(action)
                if action.action == ActionType.RETRAIN:
                    retrained += 1
                elif action.action == ActionType.DEMOTE_STRATEGY:
                    demoted += 1

        # 2. Check degradation
        for sid, s in self._strategies.items():
            if s.degradation_pct > cfg.sharpe_degradation_trigger_pct:
                triggers.append(TriggerType.DEGRADATION)
                actions.append(LoopAction(
                    ActionType.FLAG_FOR_REVIEW, TriggerType.DEGRADATION, sid,
                    f"Sharpe degraded {s.degradation_pct:.0f}%", ts,
                ))
                flagged += 1

        # 3. Refresh surrogate if enough new data
        if self._new_backtests >= cfg.min_new_backtests_for_refresh:
            triggers.append(TriggerType.DATA_THRESHOLD)
            if self.surrogate_refresh_fn:
                self.surrogate_refresh_fn()
            surrogate_refreshed = True
            self._new_backtests = 0
            self._last_surrogate_refresh = ts
            actions.append(LoopAction(
                ActionType.REFRESH_SURROGATE, TriggerType.DATA_THRESHOLD, None,
                f"Refreshed surrogate with new data", ts,
            ))

        # 4. Adapt acquisition (explore vs exploit)
        self._adapt_acquisition(ts, actions)

        # 5. Prune stale hypotheses
        pruned = self._prune_hypotheses(ts, actions)

        # 6. Save state
        result = CycleResult(
            cycle_id=self._cycle_count,
            timestamp=ts,
            triggers_fired=list(set(triggers)),
            actions_taken=actions,
            strategies_retrained=retrained,
            strategies_demoted=demoted,
            strategies_flagged=flagged,
            surrogate_refreshed=surrogate_refreshed,
            hypotheses_pruned=pruned,
            acquisition_kappa=self._kappa,
            elapsed_seconds=time.time() - start,
        )
        self._history.append(result)

        if cfg.state_dir:
            self._save_state(result)

        if cfg.verbose:
            _log(result.summary())

        return result

    # ------------------------------------------------------------------
    # DRIFT HANDLING
    # ------------------------------------------------------------------
    def _handle_drift(self, s: StrategyState, ts: str) -> LoopAction:
        """Decide what to do when drift is detected."""
        # Severe degradation -> demote
        if s.degradation_pct > 60:
            if self.demote_fn:
                self.demote_fn(s.strategy_id)
            return LoopAction(
                ActionType.DEMOTE_STRATEGY, TriggerType.DRIFT, s.strategy_id,
                f"Demoted: {s.degradation_pct:.0f}% Sharpe degradation + drift", ts,
            )

        # Moderate -> retrain
        if self.retrain_fn:
            self.retrain_fn(s.strategy_id)
        s.last_retrain = ts
        s.drift_detected = False  # Reset
        return LoopAction(
            ActionType.RETRAIN, TriggerType.DRIFT, s.strategy_id,
            f"Retrained: drift detected (degrad={s.degradation_pct:.0f}%)", ts,
        )

    # ------------------------------------------------------------------
    # ACQUISITION ADAPTATION
    # ------------------------------------------------------------------
    def _adapt_acquisition(self, ts: str, actions: List[LoopAction]):
        """
        Adjust exploration vs exploitation based on recent performance.
        If strategies are improving -> exploit more.
        If strategies are degrading -> explore more.
        """
        cfg = self.config
        if not self._strategies:
            return

        avg_degrad = np.mean([s.degradation_pct for s in self._strategies.values()])

        # More degradation -> more exploration (higher kappa)
        if avg_degrad > 20:
            self._kappa = min(self._kappa + cfg.acquisition_learning_rate, 4.0)
        elif avg_degrad < 5:
            self._kappa = max(self._kappa - cfg.acquisition_learning_rate, 0.5)

        actions.append(LoopAction(
            ActionType.ADJUST_ACQUISITION, TriggerType.SCHEDULED, None,
            f"κ -> {self._kappa:.3f} (avg_degrad={avg_degrad:.1f}%)", ts,
        ))

    # ------------------------------------------------------------------
    # HYPOTHESIS PRUNING
    # ------------------------------------------------------------------
    def _prune_hypotheses(self, ts: str, actions: List[LoopAction]) -> int:
        """Remove hypotheses that consistently underperform."""
        cfg = self.config

        # Group strategies by hypothesis
        hyp_groups: Dict[str, List[StrategyState]] = {}
        for s in self._strategies.values():
            if s.hypothesis_id:
                hyp_groups.setdefault(s.hypothesis_id, []).append(s)

        pruned = 0
        for hyp_id, strats in hyp_groups.items():
            if len(strats) < 3:
                continue

            # Check improvement rate
            improved = sum(1 for s in strats
                          if s.live_sharpe > s.backtest_sharpe * 0.8)
            rate = improved / len(strats)

            if rate < cfg.hypothesis_min_improvement_rate:
                pruned += 1
                actions.append(LoopAction(
                    ActionType.PRUNE_HYPOTHESIS, TriggerType.SCHEDULED, None,
                    f"Pruned hypothesis {hyp_id}: improvement rate={rate:.0%} "
                    f"({improved}/{len(strats)})", ts,
                ))

        return pruned

    # ------------------------------------------------------------------
    # MUTATION EFFECTIVENESS
    # ------------------------------------------------------------------
    def get_mutation_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Analyze which mutation types produce the best outcomes."""
        by_type: Dict[str, List[float]] = {}
        for s in self._strategies.values():
            if s.mutation_type and s.parent_id:
                parent = self._strategies.get(s.parent_id)
                if parent:
                    improvement = s.live_sharpe - parent.live_sharpe
                    by_type.setdefault(s.mutation_type, []).append(improvement)

        result = {}
        for mt, improvements in by_type.items():
            arr = np.array(improvements)
            result[mt] = {
                "count": len(arr),
                "avg_improvement": float(np.mean(arr)),
                "success_rate": float(np.mean(arr > 0)),
                "best": float(np.max(arr)) if len(arr) > 0 else 0,
                "worst": float(np.min(arr)) if len(arr) > 0 else 0,
            }
        return dict(sorted(result.items(), key=lambda x: -x[1]["avg_improvement"]))

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def _save_state(self, result: CycleResult):
        d = Path(self.config.state_dir)
        d.mkdir(parents=True, exist_ok=True)
        data = {
            "cycle_id": result.cycle_id,
            "timestamp": result.timestamp,
            "kappa": self._kappa,
            "n_strategies": len(self._strategies),
            "actions": [
                {"action": a.action.value, "trigger": a.trigger.value,
                 "strategy": a.strategy_id, "details": a.details}
                for a in result.actions_taken
            ],
        }
        with open(d / f"cycle_{result.cycle_id}.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------
    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def kappa(self) -> float:
        return self._kappa

    def get_history(self) -> List[CycleResult]:
        return self._history

    def get_strategy_states(self) -> Dict[str, StrategyState]:
        return dict(self._strategies)
