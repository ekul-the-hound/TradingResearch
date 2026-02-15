# ==============================================================================
# lineage_tracker.py
# ==============================================================================
# Module 1 of 4 — Phase 1: Foundation Completion
#
# Strategy Lineage Tracking System
#
# Tracks strategy genealogy: parent → child relationships, mutation types,
# generation counts, and performance metrics over time. Dual-writes to
# SQLite (fast local queries, family tree traversal) and MLflow (experiment
# tracking dashboard, artifact logging, visual lineage browsing).
#
# GitHub repo: mlflow/mlflow (https://github.com/mlflow/mlflow)
#   - Every strategy variant logged as an MLflow run
#   - Parent/child via mlflow.parentRunId tag
#   - Generation + mutation_type as run tags
#   - Backtest metrics logged as MLflow metrics
#
# Consumed by:
#   - filtering_pipeline.py (reads strategies, updates status)
#   - diversification_filter.py (updates status on survivors)
#   - Phase 2 genetic operators (register mutations, log backtests)
#
# Usage:
#     from lineage_tracker import LineageTracker
#
#     tracker = LineageTracker()
#
#     root = tracker.register_strategy(
#         name="SMA_Cross_v1", origin="discovered",
#         hypothesis="MA crossover captures trend momentum"
#     )
#     child = tracker.register_strategy(
#         name="SMA_Cross_v1_RSI", origin="mutation",
#         parent_id=root, mutation_type="add_indicator",
#         mutation_params={"indicator": "RSI", "period": 14}
#     )
#     tracker.log_backtest(child, {"sharpe_ratio": 1.2, "max_drawdown_pct": 12.5})
#     tracker.print_family_tree(root)
#
# ==============================================================================

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import config as _cfg
    _BASE_DIR = _cfg.BASE_DIR
except ImportError:
    _BASE_DIR = Path(__file__).parent


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class StrategyRecord:
    """One node in the lineage graph."""
    strategy_id: str
    name: str
    origin: str                           # discovered | mutation | crossover | manual | genetic
    parent_id: Optional[str] = None
    generation: int = 0
    mutation_type: Optional[str] = None   # add_indicator | change_params | add_filter | ...
    mutation_params: Optional[Dict] = None
    code_hash: Optional[str] = None
    hypothesis: Optional[str] = None
    created_at: str = ""
    status: str = "pending"               # pending | backtested | filtered | promoted | retired
    mlflow_run_id: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class FamilyNode:
    """Recursive tree node for visualization."""
    strategy_id: str
    name: str
    generation: int
    origin: str
    status: str
    mutation_type: Optional[str]
    sharpe_ratio: Optional[float] = None
    total_return_pct: Optional[float] = None
    children: List["FamilyNode"] = field(default_factory=list)


# ==============================================================================
# LINEAGE TRACKER
# ==============================================================================

class LineageTracker:
    """
    Strategy genealogy tracker backed by SQLite + optional MLflow.

    SQLite: fast local queries, family tree traversal, mutation stats.
    MLflow: experiment dashboard, artifact logging, visual browsing.
    """

    EXPERIMENT_NAME = "TradingLab_Lineage"

    def __init__(
        self,
        db_path: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        enable_mlflow: bool = True,
    ):
        if db_path is None:
            db_dir = _BASE_DIR / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "lineage.db")

        self.db_path = db_path
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self._init_database()

        if self.enable_mlflow:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            else:
                mlflow_dir = Path(self.db_path).parent / "mlruns"
                mlflow.set_tracking_uri(f"file://{mlflow_dir.resolve()}")
            mlflow.set_experiment(self.EXPERIMENT_NAME)
            self._client = MlflowClient()

        print(f"✓ LineageTracker initialized "
              f"(db={self.db_path}, mlflow={'ON' if self.enable_mlflow else 'OFF'})")

    # ------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id     TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                origin          TEXT NOT NULL,
                parent_id       TEXT,
                generation      INTEGER DEFAULT 0,
                mutation_type   TEXT,
                mutation_params TEXT,
                code_hash       TEXT,
                hypothesis      TEXT,
                created_at      TEXT NOT NULL,
                status          TEXT DEFAULT 'pending',
                mlflow_run_id   TEXT,
                FOREIGN KEY (parent_id) REFERENCES strategies(strategy_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS backtest_metrics (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id      TEXT NOT NULL,
                symbol           TEXT,
                timeframe        TEXT,
                sharpe_ratio     REAL,
                max_drawdown_pct REAL,
                total_return_pct REAL,
                total_trades     INTEGER,
                win_rate         REAL,
                profit_factor    REAL,
                regime_consistency REAL,
                robustness_score REAL,
                pbo_probability  REAL,
                deflated_sharpe  REAL,
                logged_at        TEXT NOT NULL,
                extra_metrics    TEXT,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            )
        """)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_lin_parent ON strategies(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_lin_origin ON strategies(origin)",
            "CREATE INDEX IF NOT EXISTS idx_lin_gen    ON strategies(generation)",
            "CREATE INDEX IF NOT EXISTS idx_lin_status ON strategies(status)",
            "CREATE INDEX IF NOT EXISTS idx_bm_sid     ON backtest_metrics(strategy_id)",
        ]:
            c.execute(idx_sql)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # REGISTER
    # ------------------------------------------------------------------
    def _generate_id(self, name: str, parent_id: Optional[str] = None) -> str:
        seed = f"{name}:{parent_id or 'root'}:{datetime.now().isoformat()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def register_strategy(
        self,
        name: str,
        origin: str,
        parent_id: Optional[str] = None,
        mutation_type: Optional[str] = None,
        mutation_params: Optional[Dict] = None,
        code_hash: Optional[str] = None,
        hypothesis: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> str:
        """Register a strategy in the lineage graph. Returns its strategy_id."""
        if strategy_id is None:
            strategy_id = self._generate_id(name, parent_id)

        generation = 0
        if parent_id:
            parent = self.get_strategy(parent_id)
            if parent:
                generation = parent.generation + 1

        rec = StrategyRecord(
            strategy_id=strategy_id, name=name, origin=origin,
            parent_id=parent_id, generation=generation,
            mutation_type=mutation_type, mutation_params=mutation_params,
            code_hash=code_hash, hypothesis=hypothesis,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO strategies
            (strategy_id, name, origin, parent_id, generation, mutation_type,
             mutation_params, code_hash, hypothesis, created_at, status, mlflow_run_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rec.strategy_id, rec.name, rec.origin, rec.parent_id,
            rec.generation, rec.mutation_type,
            json.dumps(rec.mutation_params) if rec.mutation_params else None,
            rec.code_hash, rec.hypothesis, rec.created_at, rec.status, None,
        ))
        conn.commit()
        conn.close()

        if self.enable_mlflow:
            self._log_registration_to_mlflow(rec)

        return strategy_id

    def _log_registration_to_mlflow(self, rec: StrategyRecord):
        tags = {
            "strategy_id": rec.strategy_id,
            "origin": rec.origin,
            "generation": str(rec.generation),
            "status": rec.status,
        }
        if rec.parent_id:
            parent_run = self._get_mlflow_run_id(rec.parent_id)
            if parent_run:
                tags["mlflow.parentRunId"] = parent_run
            tags["parent_strategy_id"] = rec.parent_id
        if rec.mutation_type:
            tags["mutation_type"] = rec.mutation_type

        with mlflow.start_run(run_name=rec.name, tags=tags) as run:
            mlflow.log_param("origin", rec.origin)
            mlflow.log_param("generation", rec.generation)
            if rec.mutation_params:
                for k, v in rec.mutation_params.items():
                    mlflow.log_param(f"mut_{k}", v)
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE strategies SET mlflow_run_id=? WHERE strategy_id=?",
                (run.info.run_id, rec.strategy_id),
            )
            conn.commit()
            conn.close()

    def _get_mlflow_run_id(self, strategy_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT mlflow_run_id FROM strategies WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # LOG BACKTEST
    # ------------------------------------------------------------------
    _STD_KEYS = [
        "sharpe_ratio", "max_drawdown_pct", "total_return_pct",
        "total_trades", "win_rate", "profit_factor",
        "regime_consistency", "robustness_score",
        "pbo_probability", "deflated_sharpe",
    ]

    def log_backtest(
        self,
        strategy_id: str,
        metrics: Dict[str, Any],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """Log backtest results. Returns the row id."""
        extra = {k: v for k, v in metrics.items() if k not in self._STD_KEYS}
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO backtest_metrics
            (strategy_id, symbol, timeframe, sharpe_ratio, max_drawdown_pct,
             total_return_pct, total_trades, win_rate, profit_factor,
             regime_consistency, robustness_score, pbo_probability,
             deflated_sharpe, logged_at, extra_metrics)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            strategy_id, symbol, timeframe,
            metrics.get("sharpe_ratio"), metrics.get("max_drawdown_pct"),
            metrics.get("total_return_pct"), metrics.get("total_trades"),
            metrics.get("win_rate"), metrics.get("profit_factor"),
            metrics.get("regime_consistency"), metrics.get("robustness_score"),
            metrics.get("pbo_probability"), metrics.get("deflated_sharpe"),
            datetime.now().isoformat(),
            json.dumps(extra) if extra else None,
        ))
        row_id = c.lastrowid
        conn.commit()
        conn.close()

        if self.enable_mlflow:
            run_id = self._get_mlflow_run_id(strategy_id)
            if run_id:
                for k in self._STD_KEYS:
                    v = metrics.get(k)
                    if v is not None:
                        self._client.log_metric(run_id, k, float(v))
        return row_id

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------
    def update_status(self, strategy_id: str, new_status: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE strategies SET status=? WHERE strategy_id=?",
            (new_status, strategy_id),
        )
        conn.commit()
        conn.close()
        if self.enable_mlflow:
            run_id = self._get_mlflow_run_id(strategy_id)
            if run_id:
                self._client.set_tag(run_id, "status", new_status)

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------
    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM strategies WHERE strategy_id=?", (strategy_id,)
        ).fetchone()
        conn.close()
        return self._row_to_record(row) if row else None

    def get_children(self, strategy_id: str) -> List[StrategyRecord]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM strategies WHERE parent_id=? ORDER BY created_at",
            (strategy_id,),
        ).fetchall()
        conn.close()
        return [self._row_to_record(r) for r in rows]

    def get_descendants(self, strategy_id: str) -> List[StrategyRecord]:
        out, queue = [], [strategy_id]
        while queue:
            children = self.get_children(queue.pop(0))
            out.extend(children)
            queue.extend(c.strategy_id for c in children)
        return out

    def get_best_metrics(self, strategy_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM backtest_metrics WHERE strategy_id=? ORDER BY sharpe_ratio DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_family_tree(self, root_id: str) -> Optional[FamilyNode]:
        strat = self.get_strategy(root_id)
        if not strat:
            return None
        met = self.get_best_metrics(root_id)
        node = FamilyNode(
            strategy_id=strat.strategy_id, name=strat.name,
            generation=strat.generation, origin=strat.origin,
            status=strat.status, mutation_type=strat.mutation_type,
            sharpe_ratio=met.get("sharpe_ratio") if met else None,
            total_return_pct=met.get("total_return_pct") if met else None,
        )
        for child in self.get_children(root_id):
            child_node = self.get_family_tree(child.strategy_id)
            if child_node:
                node.children.append(child_node)
        return node

    def get_all_strategies(
        self, origin: Optional[str] = None, status: Optional[str] = None,
        min_generation: int = 0, max_generation: Optional[int] = None,
    ) -> List[StrategyRecord]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        q = "SELECT * FROM strategies WHERE generation >= ?"
        p: list = [min_generation]
        if origin:
            q += " AND origin=?"; p.append(origin)
        if status:
            q += " AND status=?"; p.append(status)
        if max_generation is not None:
            q += " AND generation<=?"; p.append(max_generation)
        q += " ORDER BY generation, created_at"
        rows = conn.execute(q, p).fetchall()
        conn.close()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------
    def get_mutation_success_rates(self) -> Dict[str, Dict[str, float]]:
        """Which mutation types produce the best strategies?"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT s.mutation_type,
                   COUNT(DISTINCT s.strategy_id) AS cnt,
                   AVG(bm.sharpe_ratio)          AS avg_sharpe,
                   AVG(bm.total_return_pct)      AS avg_return,
                   SUM(CASE WHEN s.status='promoted' THEN 1 ELSE 0 END) AS promoted,
                   AVG(s.generation)              AS avg_gen
            FROM strategies s
            LEFT JOIN backtest_metrics bm ON s.strategy_id = bm.strategy_id
            WHERE s.mutation_type IS NOT NULL
            GROUP BY s.mutation_type ORDER BY avg_sharpe DESC
        """).fetchall()
        conn.close()
        return {
            r["mutation_type"]: {
                "count": r["cnt"],
                "avg_sharpe": r["avg_sharpe"] or 0.0,
                "avg_return": r["avg_return"] or 0.0,
                "promoted_pct": (r["promoted"] / r["cnt"] * 100) if r["cnt"] else 0.0,
                "avg_generation": r["avg_gen"] or 0.0,
            } for r in rows
        }

    def get_generation_stats(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT s.generation,
                   COUNT(DISTINCT s.strategy_id) AS cnt,
                   AVG(bm.sharpe_ratio) AS avg_sharpe,
                   MAX(bm.sharpe_ratio) AS best_sharpe,
                   AVG(bm.total_return_pct) AS avg_return,
                   AVG(bm.max_drawdown_pct) AS avg_dd
            FROM strategies s
            LEFT JOIN backtest_metrics bm ON s.strategy_id = bm.strategy_id
            GROUP BY s.generation ORDER BY s.generation
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_lineage_summary(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        r = conn.execute("""
            SELECT COUNT(*),
                   COUNT(DISTINCT parent_id),
                   MAX(generation),
                   SUM(CASE WHEN origin='discovered' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN origin='mutation'   THEN 1 ELSE 0 END),
                   SUM(CASE WHEN origin='crossover'  THEN 1 ELSE 0 END),
                   SUM(CASE WHEN origin='genetic'    THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status='promoted'   THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status='retired'    THEN 1 ELSE 0 END)
            FROM strategies
        """).fetchone()
        conn.close()
        return dict(zip(
            ["total_strategies","unique_parents","max_generation",
             "discovered","mutated","crossover","genetic","promoted","retired"],
            [r[i] if r[i] is not None else 0 for i in range(9)]
        ))

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    def print_family_tree(self, root_id: str, indent: int = 0):
        tree = self.get_family_tree(root_id)
        if not tree:
            print(f"  Strategy {root_id} not found."); return
        self._print_node(tree, indent)

    def _print_node(self, n: FamilyNode, indent: int):
        pre = "  " * indent + ("├── " if indent > 0 else "")
        sr = f"Sharpe={n.sharpe_ratio:.2f}" if n.sharpe_ratio is not None else "no backtest"
        icon = {"promoted":"🟢","retired":"🔴","filtered":"🟡"}.get(n.status, "⚪")
        mut = f" [{n.mutation_type}]" if n.mutation_type else ""
        print(f"{pre}{icon} {n.name} (gen {n.generation}, {sr}){mut}")
        for c in n.children:
            self._print_node(c, indent + 1)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _row_to_record(self, row) -> StrategyRecord:
        return StrategyRecord(
            strategy_id=row["strategy_id"], name=row["name"], origin=row["origin"],
            parent_id=row["parent_id"], generation=row["generation"],
            mutation_type=row["mutation_type"],
            mutation_params=json.loads(row["mutation_params"]) if row["mutation_params"] else None,
            code_hash=row["code_hash"], hypothesis=row["hypothesis"],
            created_at=row["created_at"], status=row["status"],
            mlflow_run_id=row["mlflow_run_id"],
        )

    def strategy_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
        conn.close()
        return n
