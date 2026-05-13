# ==============================================================================
# decay_calculator.py
# ==============================================================================
# Strategy Edge Decay Metrics
#
# Tracks how a strategy's performance evolves by comparing a historical
# Baseline window (first 50% of trades) against a Recent rolling window
# (last 20% of trades).
#
# Each metric is normalized to a 0-110 scale where 100 == identical to
# baseline. A composite "edge decay score" is the average of three core
# metrics (win rate, expectancy, trade frequency).
#
# Status thresholds (composite score):
#     >= 90  -> excellent
#     >= 70  -> good
#     >= 50  -> warning
#     <  50  -> poor (strategy edge likely gone)
#
# Adapted for TradingLab from the platform-agnostic spec. Unlike the
# original Flask + JS implementation, this one is backtest-first:
#   - Each backtest run produces one decay snapshot
#   - Multiple snapshots over time form the decay history
#   - No daily scheduler needed (backtests are explicit operations)
#
# Usage:
#     from decay_calculator import DecayCalculator
#
#     dc = DecayCalculator()
#
#     # Persist trades from a backtest result
#     dc.save_trades(strategy_id='rsi_mean_rev', symbol='EUR-USD',
#                    backtest_id=42, trades=result['trades'])
#
#     # Generate a snapshot from those trades
#     snap = dc.generate_snapshot(strategy_id='rsi_mean_rev',
#                                 symbol='EUR-USD')
#     print(f"Composite decay score: {snap['decay_score_composite']:.1f}")
#
#     # Retrieve history
#     history = dc.get_snapshots('rsi_mean_rev')
#
# ==============================================================================

import sqlite3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterable

# ------------------------------------------------------------------------------
# Paths (mirrors react_dashboard2.py and database.py conventions)
# ------------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

try:
    import config as _cfg
    _BASE_DIR = Path(_cfg.BASE_DIR)
    _DB_PATH = str(_cfg.DATABASE_PATH)
except Exception:
    _BASE_DIR = _THIS_DIR
    _DB_PATH = str(_BASE_DIR / "results" / "backtest_results.db")


# ------------------------------------------------------------------------------
# Tunable constants (match the friend's spec)
# ------------------------------------------------------------------------------
BASELINE_FRAC = 0.50          # First 50% of trades
RECENT_FRAC = 0.20            # Last 20% of trades
BASELINE_MIN_SOFT = 100       # Soft minimum (warning if below)
RECENT_MIN_SOFT = 50          # Soft minimum (warning if below)
HARD_MIN_TOTAL_TRADES = 20    # Hard minimum; below this we refuse to compute

# Status thresholds (composite score)
STATUS_EXCELLENT = 90.0
STATUS_GOOD = 70.0
STATUS_WARNING = 50.0

# Composite weights (equal average of the three core metrics)
COMPOSITE_METRICS = ("win_rate", "expectancy", "trade_frequency")


# ==============================================================================
# DECAY CALCULATOR
# ==============================================================================

class DecayCalculator:
    """
    Edge-decay metric engine.

    Stores two tables in the backtest results SQLite DB:
        strategy_trades            -- individual trade records
        strategy_decay_snapshots   -- per-snapshot baseline/rolling/scores
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --------------------------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS strategy_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_time TEXT,
                exit_time TEXT NOT NULL,
                pnl REAL NOT NULL,
                pnlcomm REAL,
                size REAL,
                is_long INTEGER,
                return_pct REAL,
                duration_hours REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_strat_sym_time
            ON strategy_trades(strategy_id, symbol, exit_time)
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS strategy_decay_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                total_trades INTEGER NOT NULL,

                baseline_trade_count INTEGER NOT NULL,
                baseline_win_rate REAL,
                baseline_expectancy REAL,
                baseline_trade_frequency REAL,
                baseline_profit_factor REAL,
                baseline_win_loss_ratio REAL,
                baseline_max_consecutive_losses INTEGER,
                baseline_avg_trade_duration_hours REAL,

                rolling_trade_count INTEGER NOT NULL,
                rolling_win_rate REAL,
                rolling_expectancy REAL,
                rolling_trade_frequency REAL,
                rolling_profit_factor REAL,
                rolling_win_loss_ratio REAL,
                rolling_max_consecutive_losses INTEGER,
                rolling_avg_trade_duration_hours REAL,

                decay_score_composite REAL,
                decay_score_win_rate REAL,
                decay_score_expectancy REAL,
                decay_score_trade_frequency REAL,
                decay_score_profit_factor REAL,
                decay_score_win_loss_ratio REAL,
                decay_score_max_consecutive_losses REAL,
                decay_score_avg_trade_duration REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_id, symbol, snapshot_date)
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_decay_strategy_symbol
            ON strategy_decay_snapshots(strategy_id, symbol, snapshot_date DESC)
        ''')

        conn.commit()
        conn.close()

    # --------------------------------------------------------------------------
    # TRADE PERSISTENCE
    # --------------------------------------------------------------------------
    def save_trades(
        self,
        strategy_id: str,
        symbol: str,
        trades: List[Dict[str, Any]],
        backtest_id: Optional[int] = None,
        replace: bool = False,
    ) -> int:
        """
        Persist a list of trades for a strategy/symbol pair.

        `trades` is expected to follow the shape produced by TradeRecorder in
        backtester_multi_timeframe.py:
            entry_date, exit_date, entry_price, pnl, pnlcomm, return_pct,
            size, duration_bars, is_long

        Returns:
            Number of trades inserted.
        """
        if not trades:
            return 0

        conn = self._get_conn()
        c = conn.cursor()

        if replace:
            c.execute(
                "DELETE FROM strategy_trades WHERE strategy_id = ? AND symbol = ?",
                (strategy_id, symbol),
            )

        rows = []
        for t in trades:
            entry_time = _as_iso(t.get("entry_date"))
            exit_time = _as_iso(t.get("exit_date"))
            if exit_time is None:
                # Cannot use a trade with no exit timestamp
                continue
            duration_h = None
            if entry_time and exit_time:
                try:
                    duration_h = (
                        datetime.fromisoformat(exit_time)
                        - datetime.fromisoformat(entry_time)
                    ).total_seconds() / 3600.0
                except Exception:
                    duration_h = None

            pnl = t.get("pnl")
            if pnl is None:
                pnl = t.get("pnlcomm", 0.0)

            rows.append((
                backtest_id,
                strategy_id,
                symbol,
                entry_time,
                exit_time,
                float(pnl) if pnl is not None else 0.0,
                float(t.get("pnlcomm")) if t.get("pnlcomm") is not None else None,
                float(t.get("size")) if t.get("size") is not None else None,
                int(bool(t.get("is_long"))) if t.get("is_long") is not None else None,
                float(t.get("return_pct")) if t.get("return_pct") is not None else None,
                duration_h,
            ))

        c.executemany('''
            INSERT INTO strategy_trades
            (backtest_id, strategy_id, symbol, entry_time, exit_time,
             pnl, pnlcomm, size, is_long, return_pct, duration_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()
        conn.close()
        return len(rows)

    def get_trades(
        self,
        strategy_id: str,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all persisted trades for a strategy (optionally one symbol)."""
        conn = self._get_conn()
        if symbol is not None:
            cur = conn.execute(
                "SELECT * FROM strategy_trades "
                "WHERE strategy_id = ? AND symbol = ? "
                "ORDER BY exit_time ASC",
                (strategy_id, symbol),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM strategy_trades WHERE strategy_id = ? "
                "ORDER BY exit_time ASC",
                (strategy_id,),
            )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def list_strategies(self) -> List[Dict[str, Any]]:
        """List every (strategy_id, symbol) pair that has stored trades."""
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT strategy_id, symbol, COUNT(*) AS n_trades,
                   MIN(exit_time) AS first_trade,
                   MAX(exit_time) AS last_trade
            FROM strategy_trades
            GROUP BY strategy_id, symbol
            ORDER BY strategy_id, symbol
        ''')
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    # --------------------------------------------------------------------------
    # METRIC CALCULATION
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute the seven raw metrics for a trade window.

        Expected trade keys: pnl, exit_time, [entry_time], [duration_hours]
        Trades should already be sorted by exit_time (ascending).
        """
        n = len(trades)
        if n == 0:
            return {
                "trade_count": 0, "win_rate": None, "expectancy": None,
                "trade_frequency": None, "profit_factor": None,
                "win_loss_ratio": None, "max_consecutive_losses": 0,
                "avg_trade_duration_hours": None,
            }

        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        # Win rate (percentage, 0-100)
        win_rate = (len(wins) / n) * 100.0

        # Sums + averages
        sum_wins = sum(wins) if wins else 0.0
        sum_losses = sum(losses) if losses else 0.0
        avg_win = (sum_wins / len(wins)) if wins else 0.0
        avg_loss = (sum_losses / len(losses)) if losses else 0.0

        # Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
        loss_rate = 1.0 - (win_rate / 100.0)
        expectancy = (win_rate / 100.0) * avg_win - loss_rate * avg_loss

        # Profit factor: inf when no losses, 0 when no wins (avoids div0)
        if sum_losses > 0:
            profit_factor = sum_wins / sum_losses
        else:
            profit_factor = float("inf") if sum_wins > 0 else 0.0

        # Win/loss ratio
        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float("inf") if avg_win > 0 else 0.0

        # Trade frequency: trades per day across the window
        first_ts = _parse_dt(trades[0].get("exit_time"))
        last_ts = _parse_dt(trades[-1].get("exit_time"))
        if first_ts and last_ts and last_ts > first_ts:
            span_days = (last_ts - first_ts).total_seconds() / 86400.0
            trade_frequency = n / max(span_days, 1.0 / 24.0)
        else:
            trade_frequency = None

        # Max consecutive losses
        max_streak = 0
        cur_streak = 0
        for p in pnls:
            if p < 0:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0

        # Avg trade duration (hours)
        durations = []
        for t in trades:
            d = t.get("duration_hours")
            if d is None:
                ent = _parse_dt(t.get("entry_time"))
                ext = _parse_dt(t.get("exit_time"))
                if ent and ext:
                    d = (ext - ent).total_seconds() / 3600.0
            if d is not None and d >= 0:
                durations.append(float(d))
        avg_duration = (sum(durations) / len(durations)) if durations else None

        return {
            "trade_count": n,
            "win_rate": win_rate,
            "expectancy": expectancy,
            "trade_frequency": trade_frequency,
            "profit_factor": profit_factor,
            "win_loss_ratio": win_loss_ratio,
            "max_consecutive_losses": max_streak,
            "avg_trade_duration_hours": avg_duration,
        }

    # --------------------------------------------------------------------------
    # SCORING (0-110 scale, 100 == baseline)
    # --------------------------------------------------------------------------
    @staticmethod
    def _score_standard(baseline: Optional[float], recent: Optional[float]) -> Optional[float]:
        """Higher is better: (recent / baseline) * 100."""
        if baseline is None or recent is None:
            return None
        if baseline == 0:
            # If baseline is zero, any positive recent is "infinite improvement"
            return 110.0 if recent > 0 else 100.0
        score = (recent / baseline) * 100.0
        return _clamp(score, 0.0, 110.0)

    @staticmethod
    def _score_inverted(baseline: Optional[float], recent: Optional[float]) -> Optional[float]:
        """Lower is better: (baseline / recent) * 100."""
        if baseline is None or recent is None:
            return None
        if recent == 0:
            return 110.0 if baseline > 0 else 100.0
        score = (baseline / recent) * 100.0
        return _clamp(score, 0.0, 110.0)

    @staticmethod
    def _score_expectancy(baseline: Optional[float], recent: Optional[float]) -> Optional[float]:
        """
        Expectancy has a special handling because it can be negative.

            baseline > 0, recent > 0  -> (recent / baseline) * 100
            baseline < 0, recent < 0  -> (baseline / recent) * 100  (smaller |neg| is better)
            baseline <= 0, recent > 0 -> 110 (significant improvement)
            baseline > 0, recent <= 0 -> (recent / baseline) * 100  (will be <= 0)
        """
        if baseline is None or recent is None:
            return None
        if baseline > 0 and recent > 0:
            return _clamp((recent / baseline) * 100.0, 0.0, 110.0)
        if baseline < 0 and recent < 0:
            return _clamp((baseline / recent) * 100.0, 0.0, 110.0)
        if baseline <= 0 and recent > 0:
            return 110.0
        # baseline > 0 and recent <= 0
        if baseline == 0:
            return 0.0
        return _clamp((recent / baseline) * 100.0, 0.0, 110.0)

    @classmethod
    def compute_decay_scores(
        cls,
        baseline: Dict[str, Any],
        rolling: Dict[str, Any],
    ) -> Dict[str, float]:
        """Score every metric and produce the composite."""
        scores = {
            "win_rate": cls._score_standard(
                baseline.get("win_rate"), rolling.get("win_rate")),
            "expectancy": cls._score_expectancy(
                baseline.get("expectancy"), rolling.get("expectancy")),
            "trade_frequency": cls._score_standard(
                baseline.get("trade_frequency"), rolling.get("trade_frequency")),
            "profit_factor": cls._score_standard(
                _finite(baseline.get("profit_factor")),
                _finite(rolling.get("profit_factor"))),
            "win_loss_ratio": cls._score_standard(
                _finite(baseline.get("win_loss_ratio")),
                _finite(rolling.get("win_loss_ratio"))),
            "max_consecutive_losses": cls._score_inverted(
                baseline.get("max_consecutive_losses"),
                rolling.get("max_consecutive_losses")),
            "avg_trade_duration": cls._score_inverted(
                baseline.get("avg_trade_duration_hours"),
                rolling.get("avg_trade_duration_hours")),
        }

        composite_vals = [scores[k] for k in COMPOSITE_METRICS if scores.get(k) is not None]
        composite = (sum(composite_vals) / len(composite_vals)) if composite_vals else None
        scores["composite"] = composite
        return scores

    # --------------------------------------------------------------------------
    # SNAPSHOT GENERATION
    # --------------------------------------------------------------------------
    def generate_snapshot(
        self,
        strategy_id: str,
        symbol: str,
        trades: Optional[List[Dict[str, Any]]] = None,
        snapshot_date: Optional[str] = None,
        persist: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute and persist one decay snapshot.

        If `trades` is None, trades are loaded from the strategy_trades table.

        Returns the snapshot dict, or None when there aren't enough trades.
        """
        if trades is None:
            trades = self.get_trades(strategy_id, symbol)
        trades = sorted(trades, key=lambda t: _parse_dt(t.get("exit_time")) or datetime.min)

        total = len(trades)
        if total < HARD_MIN_TOTAL_TRADES:
            return None

        baseline_n = max(int(total * BASELINE_FRAC), 1)
        recent_n = max(int(total * RECENT_FRAC), 1)

        baseline_window = trades[:baseline_n]
        recent_window = trades[-recent_n:]

        baseline = self.compute_metrics(baseline_window)
        rolling = self.compute_metrics(recent_window)
        scores = self.compute_decay_scores(baseline, rolling)

        snapshot_date = snapshot_date or datetime.now(timezone.utc).date().isoformat()

        snap = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "snapshot_date": snapshot_date,
            "total_trades": total,
            "baseline_trade_count": baseline["trade_count"],
            "baseline_win_rate": baseline["win_rate"],
            "baseline_expectancy": baseline["expectancy"],
            "baseline_trade_frequency": baseline["trade_frequency"],
            "baseline_profit_factor": _finite(baseline["profit_factor"]),
            "baseline_win_loss_ratio": _finite(baseline["win_loss_ratio"]),
            "baseline_max_consecutive_losses": baseline["max_consecutive_losses"],
            "baseline_avg_trade_duration_hours": baseline["avg_trade_duration_hours"],
            "rolling_trade_count": rolling["trade_count"],
            "rolling_win_rate": rolling["win_rate"],
            "rolling_expectancy": rolling["expectancy"],
            "rolling_trade_frequency": rolling["trade_frequency"],
            "rolling_profit_factor": _finite(rolling["profit_factor"]),
            "rolling_win_loss_ratio": _finite(rolling["win_loss_ratio"]),
            "rolling_max_consecutive_losses": rolling["max_consecutive_losses"],
            "rolling_avg_trade_duration_hours": rolling["avg_trade_duration_hours"],
            "decay_score_composite": scores["composite"],
            "decay_score_win_rate": scores["win_rate"],
            "decay_score_expectancy": scores["expectancy"],
            "decay_score_trade_frequency": scores["trade_frequency"],
            "decay_score_profit_factor": scores["profit_factor"],
            "decay_score_win_loss_ratio": scores["win_loss_ratio"],
            "decay_score_max_consecutive_losses": scores["max_consecutive_losses"],
            "decay_score_avg_trade_duration": scores["avg_trade_duration"],
        }

        if persist:
            self._persist_snapshot(snap)

        return snap

    def _persist_snapshot(self, snap: Dict[str, Any]):
        conn = self._get_conn()
        cols = [
            "strategy_id", "symbol", "snapshot_date", "total_trades",
            "baseline_trade_count", "baseline_win_rate", "baseline_expectancy",
            "baseline_trade_frequency", "baseline_profit_factor",
            "baseline_win_loss_ratio", "baseline_max_consecutive_losses",
            "baseline_avg_trade_duration_hours",
            "rolling_trade_count", "rolling_win_rate", "rolling_expectancy",
            "rolling_trade_frequency", "rolling_profit_factor",
            "rolling_win_loss_ratio", "rolling_max_consecutive_losses",
            "rolling_avg_trade_duration_hours",
            "decay_score_composite", "decay_score_win_rate",
            "decay_score_expectancy", "decay_score_trade_frequency",
            "decay_score_profit_factor", "decay_score_win_loss_ratio",
            "decay_score_max_consecutive_losses", "decay_score_avg_trade_duration",
        ]
        placeholders = ",".join("?" for _ in cols)
        sql = (
            f"INSERT OR REPLACE INTO strategy_decay_snapshots "
            f"({','.join(cols)}) VALUES ({placeholders})"
        )
        conn.execute(sql, tuple(snap.get(c) for c in cols))
        conn.commit()
        conn.close()

    def generate_all_snapshots(
        self,
        snapshot_date: Optional[str] = None,
    ) -> Dict[str, int]:
        """Generate a snapshot for every (strategy_id, symbol) with enough trades."""
        produced = 0
        skipped = 0
        for entry in self.list_strategies():
            snap = self.generate_snapshot(
                entry["strategy_id"], entry["symbol"],
                snapshot_date=snapshot_date,
            )
            if snap is None:
                skipped += 1
            else:
                produced += 1
        return {"produced": produced, "skipped": skipped}

    # --------------------------------------------------------------------------
    # SNAPSHOT RETRIEVAL
    # --------------------------------------------------------------------------
    def get_snapshots(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve decay snapshots; newest first."""
        conn = self._get_conn()
        sql = "SELECT * FROM strategy_decay_snapshots WHERE 1=1"
        params: List[Any] = []
        if strategy_id:
            sql += " AND strategy_id = ?"; params.append(strategy_id)
        if symbol:
            sql += " AND symbol = ?"; params.append(symbol)
        sql += " ORDER BY snapshot_date DESC, created_at DESC"
        if limit:
            sql += " LIMIT ?"; params.append(int(limit))
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        conn.close()
        return rows

    def latest_per_strategy(self) -> List[Dict[str, Any]]:
        """One row per (strategy_id, symbol) -- the newest snapshot only."""
        conn = self._get_conn()
        cur = conn.execute('''
            SELECT s.* FROM strategy_decay_snapshots s
            INNER JOIN (
                SELECT strategy_id, symbol, MAX(snapshot_date) AS d
                FROM strategy_decay_snapshots
                GROUP BY strategy_id, symbol
            ) latest
            ON s.strategy_id = latest.strategy_id
               AND s.symbol = latest.symbol
               AND s.snapshot_date = latest.d
            ORDER BY s.decay_score_composite ASC
        ''')
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def clear_decay_data(self, strategy_id: str, symbol: Optional[str] = None):
        """Delete decay snapshots (does not touch raw trades)."""
        conn = self._get_conn()
        if symbol is not None:
            conn.execute(
                "DELETE FROM strategy_decay_snapshots "
                "WHERE strategy_id = ? AND symbol = ?",
                (strategy_id, symbol),
            )
        else:
            conn.execute(
                "DELETE FROM strategy_decay_snapshots WHERE strategy_id = ?",
                (strategy_id,),
            )
        conn.commit()
        conn.close()

    # --------------------------------------------------------------------------
    # STATUS CLASSIFICATION
    # --------------------------------------------------------------------------
    @staticmethod
    def classify_status(composite: Optional[float]) -> str:
        """Return one of: 'excellent', 'good', 'warning', 'poor', 'unknown'."""
        if composite is None:
            return "unknown"
        if composite >= STATUS_EXCELLENT: return "excellent"
        if composite >= STATUS_GOOD:      return "good"
        if composite >= STATUS_WARNING:   return "warning"
        return "poor"


# ==============================================================================
# UTILITIES
# ==============================================================================

def _as_iso(value: Any) -> Optional[str]:
    """Coerce a datetime-like into ISO 8601 text, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        # pandas Timestamp etc.
        return value.isoformat()  # type: ignore[attr-defined]
    except Exception:
        return str(value)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    pass
    try:
        return value.to_pydatetime()  # type: ignore[attr-defined]
    except Exception:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _finite(x: Optional[float]) -> Optional[float]:
    """Coerce inf/nan to None so we don't poison scoring math."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if xf != xf or xf == float("inf") or xf == float("-inf"):
        return None
    return xf


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TradingLab Edge Decay Calculator")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List strategies with stored trades")
    p_gen = sub.add_parser("generate", help="Generate a snapshot")
    p_gen.add_argument("--strategy", required=True)
    p_gen.add_argument("--symbol", required=True)
    p_gen_all = sub.add_parser("generate-all", help="Snapshot every strategy")
    p_show = sub.add_parser("show", help="Show snapshots for a strategy")
    p_show.add_argument("--strategy", required=True)
    p_show.add_argument("--symbol", default=None)

    args = parser.parse_args()
    dc = DecayCalculator()

    if args.cmd == "list":
        for r in dc.list_strategies():
            print(f"  {r['strategy_id']:30s} {r['symbol']:12s} "
                  f"trades={r['n_trades']:5d}  "
                  f"{r['first_trade']} -> {r['last_trade']}")
    elif args.cmd == "generate":
        snap = dc.generate_snapshot(args.strategy, args.symbol)
        if snap is None:
            print(f"Not enough trades (need >= {HARD_MIN_TOTAL_TRADES})")
        else:
            print(json.dumps(snap, indent=2, default=str))
    elif args.cmd == "generate-all":
        res = dc.generate_all_snapshots()
        print(f"Produced: {res['produced']}, Skipped: {res['skipped']}")
    elif args.cmd == "show":
        snaps = dc.get_snapshots(args.strategy, args.symbol)
        for s in snaps:
            print(f"  {s['snapshot_date']}  {s['symbol']:12s}  "
                  f"composite={s['decay_score_composite']:.1f}  "
                  f"status={DecayCalculator.classify_status(s['decay_score_composite'])}")
    else:
        parser.print_help()
