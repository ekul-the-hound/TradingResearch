# ==============================================================================
# retraining_scheduler.py
# ==============================================================================
# Phase 6, Module 3: Retraining Scheduler
#
# Orchestrates WHEN and HOW strategies get retrained. Supports:
#   1. Rolling window: fixed lookback, slides forward
#   2. Expanding window: grows as data accumulates
#   3. Triggered: drift/degradation fires immediate retrain
#   4. Scheduled: periodic (daily/weekly/monthly) maintenance
#   5. Adaptive: frequency adjusts based on regime volatility
#
# Each retraining job:
#   a. Selects data window (rolling/expanding)
#   b. Re-runs backtest on new data
#   c. Compares new vs old parameters
#   d. Decides: keep old, adopt new, or flag for review
#   e. Logs everything to experiment_tracker
#
# Consumed by:
#   - learning_loop.py (triggers retrain jobs)
#   - strategy_lifecycle.py (retrain before promotion)
#   - Main scheduler / cron
#
# Usage:
#     from retraining_scheduler import RetrainingScheduler, ScheduleConfig
#     sched = RetrainingScheduler(config=ScheduleConfig())
#     sched.register("S_001", data_start="2023-01-01", data_end="2025-01-01")
#     jobs = sched.get_due_jobs()
#     for job in jobs:
#         result = sched.execute_job(job)
# ==============================================================================

import numpy as np
import json
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class WindowType(Enum):
    ROLLING = "rolling"
    EXPANDING = "expanding"
    ANCHORED = "anchored"      # Fixed start, expanding end


class RetrainTrigger(Enum):
    SCHEDULED = "scheduled"
    DRIFT = "drift"
    DEGRADATION = "degradation"
    MANUAL = "manual"
    PROMOTION_GATE = "promotion_gate"


class RetrainDecision(Enum):
    ADOPT_NEW = "adopt_new"       # New params are better
    KEEP_OLD = "keep_old"         # Old params still win
    FLAG_REVIEW = "flag_review"   # Ambiguous, needs human review
    ABORT = "abort"               # Retrain failed


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ScheduleConfig:
    """Retraining schedule configuration."""
    # Window
    window_type: WindowType = WindowType.ROLLING
    rolling_window_days: int = 252         # 1 year lookback
    min_window_days: int = 63              # 3 months minimum
    step_days: int = 21                    # Slide by 1 month

    # Schedule
    periodic_interval_days: int = 7        # Weekly by default
    max_concurrent_jobs: int = 5

    # Decision thresholds
    min_improvement_pct: float = 5.0       # New must be 5%+ better
    max_degradation_pct: float = 10.0      # Abort if new is 10%+ worse
    min_backtest_trades: int = 30          # Minimum trades for valid backtest

    # Adaptive schedule
    adaptive: bool = True
    high_vol_interval_days: int = 3        # Retrain more often in high vol
    low_vol_interval_days: int = 14        # Retrain less in low vol
    vol_regime_threshold: float = 1.5      # Vol > 1.5x normal = high vol

    # Persistence
    state_dir: str = "data/retraining"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class DataWindow:
    """Time window for retraining data."""
    start: str          # ISO date
    end: str            # ISO date
    n_days: int
    window_type: WindowType


@dataclass
class RetrainJob:
    """A single retraining job."""
    job_id: str
    strategy_id: str
    trigger: RetrainTrigger
    window: DataWindow
    created_at: str
    priority: int = 0         # Higher = more urgent
    status: str = "pending"   # pending, running, complete, failed


@dataclass
class RetrainResult:
    """Output of a retraining job."""
    job_id: str
    strategy_id: str
    decision: RetrainDecision
    old_sharpe: float
    new_sharpe: float
    improvement_pct: float
    old_params: Dict[str, Any]
    new_params: Dict[str, Any]
    window: DataWindow
    elapsed_seconds: float
    message: str

    def __str__(self):
        icon = {
            RetrainDecision.ADOPT_NEW: "[OK]",
            RetrainDecision.KEEP_OLD: "⏸️",
            RetrainDecision.FLAG_REVIEW: "👁️",
            RetrainDecision.ABORT: "[FAIL]",
        }[self.decision]
        return (
            f"{icon} [{self.strategy_id}] {self.decision.value}: "
            f"Sharpe {self.old_sharpe:.3f} -> {self.new_sharpe:.3f} "
            f"({self.improvement_pct:+.1f}%)"
        )


@dataclass
class StrategySchedule:
    """Schedule state for one strategy."""
    strategy_id: str
    last_retrain: Optional[str] = None
    next_retrain: Optional[str] = None
    data_start: str = ""
    data_end: str = ""
    current_params: Dict[str, Any] = field(default_factory=dict)
    retrain_count: int = 0
    current_sharpe: float = 0.0
    current_vol: float = 0.0
    normal_vol: float = 0.0


# ==============================================================================
# RETRAINING SCHEDULER
# ==============================================================================

class RetrainingScheduler:
    """Orchestrates strategy retraining."""

    def __init__(
        self,
        config: Optional[ScheduleConfig] = None,
        backtest_fn: Optional[Callable] = None,
    ):
        self.config = config or ScheduleConfig()
        self.backtest_fn = backtest_fn
        self._schedules: Dict[str, StrategySchedule] = {}
        self._job_queue: List[RetrainJob] = []
        self._results: List[RetrainResult] = []
        self._job_counter = 0

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------
    def register(
        self,
        strategy_id: str,
        data_start: str = "2020-01-01",
        data_end: str = "",
        current_params: Optional[Dict] = None,
        current_sharpe: float = 0.0,
    ):
        if not data_end:
            data_end = datetime.now().strftime("%Y-%m-%d")
        self._schedules[strategy_id] = StrategySchedule(
            strategy_id=strategy_id,
            data_start=data_start,
            data_end=data_end,
            current_params=current_params or {},
            current_sharpe=current_sharpe,
            next_retrain=self._compute_next(None),
        )

    # ------------------------------------------------------------------
    # WINDOW COMPUTATION
    # ------------------------------------------------------------------
    def compute_window(self, strategy_id: str) -> DataWindow:
        """Compute the data window for retraining."""
        s = self._schedules[strategy_id]
        cfg = self.config
        end = datetime.now()

        if cfg.window_type == WindowType.ROLLING:
            start = end - timedelta(days=cfg.rolling_window_days)
            # Don't go before data start
            data_start = datetime.fromisoformat(s.data_start)
            start = max(start, data_start)
        elif cfg.window_type == WindowType.EXPANDING:
            start = datetime.fromisoformat(s.data_start)
        elif cfg.window_type == WindowType.ANCHORED:
            start = datetime.fromisoformat(s.data_start)
        else:
            start = end - timedelta(days=cfg.rolling_window_days)

        n_days = (end - start).days
        return DataWindow(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            n_days=n_days,
            window_type=cfg.window_type,
        )

    # ------------------------------------------------------------------
    # JOB MANAGEMENT
    # ------------------------------------------------------------------
    def get_due_jobs(self, now: Optional[str] = None) -> List[RetrainJob]:
        """Get all strategies due for retraining."""
        current = datetime.fromisoformat(now) if now else datetime.now()
        due = []

        for sid, s in self._schedules.items():
            if s.next_retrain:
                next_dt = datetime.fromisoformat(s.next_retrain)
                if current >= next_dt:
                    job = self._create_job(sid, RetrainTrigger.SCHEDULED)
                    due.append(job)

        # Sort by priority (higher first)
        return sorted(due, key=lambda j: -j.priority)

    def trigger_retrain(
        self,
        strategy_id: str,
        trigger: RetrainTrigger = RetrainTrigger.MANUAL,
        priority: int = 5,
    ) -> RetrainJob:
        """Manually trigger a retrain job."""
        job = self._create_job(strategy_id, trigger, priority)
        return job

    def _create_job(
        self, strategy_id: str, trigger: RetrainTrigger, priority: int = 0,
    ) -> RetrainJob:
        self._job_counter += 1
        window = self.compute_window(strategy_id)
        job = RetrainJob(
            job_id=f"RJ_{self._job_counter:04d}",
            strategy_id=strategy_id,
            trigger=trigger,
            window=window,
            created_at=datetime.now().isoformat(),
            priority=priority,
        )
        self._job_queue.append(job)
        return job

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    def execute_job(self, job: RetrainJob) -> RetrainResult:
        """Execute a retraining job."""
        import time
        start = time.time()
        cfg = self.config
        s = self._schedules.get(job.strategy_id)

        if not s:
            return RetrainResult(
                job.job_id, job.strategy_id, RetrainDecision.ABORT,
                0, 0, 0, {}, {}, job.window, 0, "Strategy not registered",
            )

        job.status = "running"
        old_sharpe = s.current_sharpe
        old_params = dict(s.current_params)

        # Run backtest with new window
        if self.backtest_fn:
            bt_result = self.backtest_fn(
                job.strategy_id, job.window.start, job.window.end,
            )
            new_sharpe = bt_result.get("sharpe_ratio", 0)
            new_params = bt_result.get("params", old_params)
        else:
            # Simulated retrain
            np.random.seed(hash(job.job_id) % 2**31)
            noise = np.random.normal(0, 0.15)
            new_sharpe = old_sharpe + noise
            new_params = {**old_params, "retrained": True, "window": job.window.start}

        # Decision
        if old_sharpe > 0:
            improvement = (new_sharpe - old_sharpe) / abs(old_sharpe) * 100
        else:
            improvement = 100 if new_sharpe > old_sharpe else -100

        if improvement >= cfg.min_improvement_pct:
            decision = RetrainDecision.ADOPT_NEW
            s.current_sharpe = new_sharpe
            s.current_params = new_params
            msg = f"Adopted: {improvement:+.1f}% improvement"
        elif improvement <= -cfg.max_degradation_pct:
            decision = RetrainDecision.ABORT
            msg = f"Aborted: {improvement:+.1f}% degradation"
        elif abs(improvement) < cfg.min_improvement_pct:
            decision = RetrainDecision.KEEP_OLD
            msg = f"Kept old: {improvement:+.1f}% (below threshold)"
        else:
            decision = RetrainDecision.FLAG_REVIEW
            msg = f"Flagged: {improvement:+.1f}% (ambiguous)"

        # Update schedule
        s.retrain_count += 1
        s.last_retrain = datetime.now().isoformat()
        s.next_retrain = self._compute_next(s)
        job.status = "complete"

        result = RetrainResult(
            job_id=job.job_id,
            strategy_id=job.strategy_id,
            decision=decision,
            old_sharpe=old_sharpe,
            new_sharpe=new_sharpe,
            improvement_pct=improvement,
            old_params=old_params,
            new_params=new_params,
            window=job.window,
            elapsed_seconds=time.time() - start,
            message=msg,
        )
        self._results.append(result)

        if cfg.state_dir:
            self._save_result(result)

        return result

    # ------------------------------------------------------------------
    # ADAPTIVE SCHEDULING
    # ------------------------------------------------------------------
    def update_volatility(self, strategy_id: str, current_vol: float, normal_vol: float):
        """Update vol for adaptive scheduling."""
        if strategy_id in self._schedules:
            s = self._schedules[strategy_id]
            s.current_vol = current_vol
            s.normal_vol = normal_vol
            # Recompute next retrain
            s.next_retrain = self._compute_next(s)

    def _compute_next(self, s: Optional[StrategySchedule]) -> str:
        cfg = self.config
        base = datetime.now()

        if cfg.adaptive and s and s.normal_vol > 0:
            vol_ratio = s.current_vol / s.normal_vol
            if vol_ratio > cfg.vol_regime_threshold:
                interval = cfg.high_vol_interval_days
            elif vol_ratio < 1 / cfg.vol_regime_threshold:
                interval = cfg.low_vol_interval_days
            else:
                interval = cfg.periodic_interval_days
        else:
            interval = cfg.periodic_interval_days

        return (base + timedelta(days=interval)).isoformat()

    # ------------------------------------------------------------------
    # WALK-FORWARD RETRAIN
    # ------------------------------------------------------------------
    def walk_forward_retrain(
        self,
        strategy_id: str,
        n_folds: int = 5,
    ) -> List[RetrainResult]:
        """
        Walk-forward retraining: slide window forward, retrain at each step.
        Returns results for each fold.
        """
        s = self._schedules.get(strategy_id)
        if not s:
            return []

        cfg = self.config
        start = datetime.fromisoformat(s.data_start)
        end = datetime.now()
        total_days = (end - start).days
        fold_size = total_days // n_folds

        results = []
        for i in range(n_folds):
            fold_start = start + timedelta(days=i * cfg.step_days)
            fold_end = fold_start + timedelta(days=min(cfg.rolling_window_days, fold_size))
            if fold_end > end:
                fold_end = end

            window = DataWindow(
                start=fold_start.strftime("%Y-%m-%d"),
                end=fold_end.strftime("%Y-%m-%d"),
                n_days=(fold_end - fold_start).days,
                window_type=WindowType.ROLLING,
            )

            self._job_counter += 1
            job = RetrainJob(
                job_id=f"WF_{self._job_counter:04d}",
                strategy_id=strategy_id,
                trigger=RetrainTrigger.SCHEDULED,
                window=window,
                created_at=datetime.now().isoformat(),
            )
            result = self.execute_job(job)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------
    def get_schedule(self, strategy_id: str) -> Optional[StrategySchedule]:
        return self._schedules.get(strategy_id)

    def get_all_schedules(self) -> Dict[str, StrategySchedule]:
        return dict(self._schedules)

    def get_results(self) -> List[RetrainResult]:
        return self._results

    def get_pending_jobs(self) -> List[RetrainJob]:
        return [j for j in self._job_queue if j.status == "pending"]

    def summary(self) -> Dict[str, Any]:
        return {
            "registered": len(self._schedules),
            "pending_jobs": len(self.get_pending_jobs()),
            "total_retrains": len(self._results),
            "adopted": sum(1 for r in self._results if r.decision == RetrainDecision.ADOPT_NEW),
            "kept_old": sum(1 for r in self._results if r.decision == RetrainDecision.KEEP_OLD),
            "aborted": sum(1 for r in self._results if r.decision == RetrainDecision.ABORT),
            "flagged": sum(1 for r in self._results if r.decision == RetrainDecision.FLAG_REVIEW),
        }

    # ------------------------------------------------------------------
    def _save_result(self, result: RetrainResult):
        d = Path(self.config.state_dir)
        d.mkdir(parents=True, exist_ok=True)
        data = {
            "job_id": result.job_id,
            "strategy_id": result.strategy_id,
            "decision": result.decision.value,
            "old_sharpe": result.old_sharpe,
            "new_sharpe": result.new_sharpe,
            "improvement_pct": result.improvement_pct,
            "window_start": result.window.start,
            "window_end": result.window.end,
            "message": result.message,
        }
        with open(d / f"{result.job_id}.json", "w") as f:
            json.dump(data, f, indent=2)
