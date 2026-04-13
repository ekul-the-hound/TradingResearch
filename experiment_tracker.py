# ==============================================================================
# experiment_tracker.py
# ==============================================================================
# Phase 6, Module 5: Experiment Tracker
#
# Persistent memory for all TradingLab experiments. Logs every backtest,
# optimization run, live performance snapshot, and retraining result.
# Enables comparison across experiments and long-term trend analysis.
#
# Inspired by MLflow but purpose-built for quant strategies:
#   - Runs: individual backtests or live snapshots
#   - Experiments: groups of related runs (e.g., one optimization sweep)
#   - Metrics: time-series of performance values
#   - Parameters: strategy configuration
#   - Tags: mutation type, hypothesis, regime, lifecycle state
#
# Storage: JSON-based file system (no external dependencies).
# Upgrade path: swap to MLflow/SQL when needed.
#
# Consumed by:
#   - learning_loop.py (log cycle results)
#   - lineage_analytics.py (query historical performance)
#   - retraining_scheduler.py (compare old vs new)
#   - dashboard (experiment browser)
#
# Usage:
#     from experiment_tracker import ExperimentTracker
#     tracker = ExperimentTracker("data/experiments")
#     exp_id = tracker.create_experiment("optimization_v2")
#     run_id = tracker.start_run(exp_id, "S_001_backtest")
#     tracker.log_params(run_id, {"sma_fast": 10, "sma_slow": 50})
#     tracker.log_metrics(run_id, {"sharpe": 1.8, "max_dd": 0.12})
#     tracker.end_run(run_id)
# ==============================================================================

import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class RunStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Run:
    """A single experimental run (backtest, live snapshot, etc)."""
    run_id: str
    experiment_id: str
    name: str
    status: RunStatus = RunStatus.RUNNING
    start_time: str = ""
    end_time: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class Experiment:
    """A group of related runs."""
    experiment_id: str
    name: str
    description: str = ""
    created_at: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    run_ids: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Output of comparing multiple runs."""
    run_ids: List[str]
    metric_name: str
    values: Dict[str, float]         # run_id -> metric value
    best_run_id: str
    best_value: float
    worst_run_id: str
    worst_value: float
    mean: float
    std: float


# ==============================================================================
# EXPERIMENT TRACKER
# ==============================================================================

class ExperimentTracker:
    """
    Persistent experiment logging and comparison.
    """

    def __init__(self, base_dir: str = "data/experiments"):
        self.base_dir = Path(base_dir)
        self._experiments: Dict[str, Experiment] = {}
        self._runs: Dict[str, Run] = {}
        self._run_counter = 0
        self._exp_counter = 0

        # Load existing state if available
        self._load_state()

    # ------------------------------------------------------------------
    # EXPERIMENTS
    # ------------------------------------------------------------------
    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        self._exp_counter += 1
        exp_id = f"EXP_{self._exp_counter:04d}"
        exp = Experiment(
            experiment_id=exp_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            tags=tags or {},
        )
        self._experiments[exp_id] = exp
        return exp_id

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        return sorted(self._experiments.values(), key=lambda e: e.created_at, reverse=True)

    # ------------------------------------------------------------------
    # RUNS
    # ------------------------------------------------------------------
    def start_run(
        self,
        experiment_id: str,
        name: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        self._run_counter += 1
        run_id = f"RUN_{self._run_counter:06d}"
        run = Run(
            run_id=run_id,
            experiment_id=experiment_id,
            name=name or run_id,
            status=RunStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            tags=tags or {},
        )
        self._runs[run_id] = run

        if experiment_id in self._experiments:
            self._experiments[experiment_id].run_ids.append(run_id)

        return run_id

    def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED):
        if run_id in self._runs:
            self._runs[run_id].status = status
            self._runs[run_id].end_time = datetime.now().isoformat()

    def fail_run(self, run_id: str, error: str = ""):
        if run_id in self._runs:
            self._runs[run_id].status = RunStatus.FAILED
            self._runs[run_id].end_time = datetime.now().isoformat()
            self._runs[run_id].tags["error"] = error

    def get_run(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------
    def log_params(self, run_id: str, params: Dict[str, Any]):
        if run_id in self._runs:
            self._runs[run_id].params.update(params)

    def log_param(self, run_id: str, key: str, value: Any):
        if run_id in self._runs:
            self._runs[run_id].params[key] = value

    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        if run_id in self._runs:
            run = self._runs[run_id]
            ts = datetime.now().isoformat()
            for k, v in metrics.items():
                run.metrics[k] = v
                run.metric_history.setdefault(k, []).append((ts, v))

    def log_metric(self, run_id: str, key: str, value: float):
        self.log_metrics(run_id, {key: value})

    def log_tags(self, run_id: str, tags: Dict[str, str]):
        if run_id in self._runs:
            self._runs[run_id].tags.update(tags)

    def log_artifact(self, run_id: str, artifact_path: str):
        if run_id in self._runs:
            self._runs[run_id].artifacts.append(artifact_path)

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------
    def search_runs(
        self,
        experiment_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, str]] = None,
        min_metric: Optional[Dict[str, float]] = None,
        max_metric: Optional[Dict[str, float]] = None,
        status: Optional[RunStatus] = None,
        order_by: str = "start_time",
        ascending: bool = False,
        limit: int = 100,
    ) -> List[Run]:
        """Search runs with filters."""
        results = list(self._runs.values())

        if experiment_id:
            results = [r for r in results if r.experiment_id == experiment_id]

        if status:
            results = [r for r in results if r.status == status]

        if filter_tags:
            for k, v in filter_tags.items():
                results = [r for r in results if r.tags.get(k) == v]

        if min_metric:
            for k, v in min_metric.items():
                results = [r for r in results if r.metrics.get(k, float("-inf")) >= v]

        if max_metric:
            for k, v in max_metric.items():
                results = [r for r in results if r.metrics.get(k, float("inf")) <= v]

        # Sort
        if order_by == "start_time":
            results.sort(key=lambda r: r.start_time, reverse=not ascending)
        elif order_by in ("sharpe", "sharpe_ratio"):
            results.sort(key=lambda r: r.metrics.get("sharpe", 0), reverse=not ascending)
        else:
            results.sort(key=lambda r: r.metrics.get(order_by, 0), reverse=not ascending)

        return results[:limit]

    def get_best_run(
        self,
        experiment_id: str,
        metric: str = "sharpe",
    ) -> Optional[Run]:
        runs = self.search_runs(experiment_id=experiment_id, order_by=metric, limit=1)
        return runs[0] if runs else None

    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------
    def compare_runs(
        self,
        run_ids: List[str],
        metric: str = "sharpe",
    ) -> Optional[ComparisonResult]:
        """Compare a metric across multiple runs."""
        values = {}
        for rid in run_ids:
            run = self._runs.get(rid)
            if run and metric in run.metrics:
                values[rid] = run.metrics[metric]

        if not values:
            return None

        import numpy as np
        vals = list(values.values())
        best_id = max(values, key=values.get)
        worst_id = min(values, key=values.get)

        return ComparisonResult(
            run_ids=list(values.keys()),
            metric_name=metric,
            values=values,
            best_run_id=best_id,
            best_value=values[best_id],
            worst_run_id=worst_id,
            worst_value=values[worst_id],
            mean=float(np.mean(vals)),
            std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
        )

    def compare_experiments(
        self,
        exp_ids: List[str],
        metric: str = "sharpe",
    ) -> Dict[str, Dict[str, float]]:
        """Compare experiments by aggregating their run metrics."""
        import numpy as np
        result = {}
        for eid in exp_ids:
            runs = self.search_runs(experiment_id=eid)
            vals = [r.metrics.get(metric, 0) for r in runs if metric in r.metrics]
            if vals:
                result[eid] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                    "best": float(np.max(vals)),
                    "n_runs": len(vals),
                }
        return result

    # ------------------------------------------------------------------
    # METRIC TRENDS
    # ------------------------------------------------------------------
    def get_metric_trend(self, run_id: str, metric: str) -> List[Tuple[str, float]]:
        """Get time series of a metric for a run."""
        run = self._runs.get(run_id)
        if run and metric in run.metric_history:
            return run.metric_history[metric]
        return []

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def save(self):
        """Save all state to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        exp_data = {
            eid: {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "description": e.description,
                "created_at": e.created_at,
                "tags": e.tags,
                "run_ids": e.run_ids,
            }
            for eid, e in self._experiments.items()
        }
        with open(self.base_dir / "experiments.json", "w") as f:
            json.dump(exp_data, f, indent=2, default=str)

        run_data = {}
        for rid, r in self._runs.items():
            run_data[rid] = {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "name": r.name,
                "status": r.status.value,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "params": r.params,
                "metrics": r.metrics,
                "tags": r.tags,
                "artifacts": r.artifacts,
            }
        with open(self.base_dir / "runs.json", "w") as f:
            json.dump(run_data, f, indent=2, default=str)

    def _load_state(self):
        """Load from disk if available."""
        exp_file = self.base_dir / "experiments.json"
        run_file = self.base_dir / "runs.json"

        if exp_file.exists():
            with open(exp_file) as f:
                data = json.load(f)
            for eid, d in data.items():
                self._experiments[eid] = Experiment(**d)
                num = int(eid.split("_")[1]) if "_" in eid else 0
                self._exp_counter = max(self._exp_counter, num)

        if run_file.exists():
            with open(run_file) as f:
                data = json.load(f)
            for rid, d in data.items():
                d["status"] = RunStatus(d["status"])
                d.pop("metric_history", None)
                self._runs[rid] = Run(**d)
                num = int(rid.split("_")[1]) if "_" in rid else 0
                self._run_counter = max(self._run_counter, num)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "experiments": len(self._experiments),
            "runs": len(self._runs),
            "completed": sum(1 for r in self._runs.values() if r.status == RunStatus.COMPLETED),
            "failed": sum(1 for r in self._runs.values() if r.status == RunStatus.FAILED),
            "running": sum(1 for r in self._runs.values() if r.status == RunStatus.RUNNING),
        }
