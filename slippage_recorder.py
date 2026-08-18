# ==============================================================================
# slippage_recorder.py -- Observed Slippage / Spread Recorder
# ==============================================================================
# Records what fills ACTUALLY cost -- signal price vs. real fill price, plus the
# quoted spread at entry -- during shadow/live trading, then aggregates those
# observations per symbol into empirical slippage and spread estimates that can
# replace the ASSUMED constants in the cost model.
#
# WHY IT EXISTS:
#   cost_adjusted_scoring.CostProfile carries slippage_pct and spread_pct as
#   fixed guesses (e.g. "0.5 pip slippage"). Those guesses are the difference
#   between a backtest that survives live and one that dies to costs it never
#   modelled. The only way to know the real numbers is to measure them on the
#   broker's own feed. This recorder is that measurement layer: it turns live
#   fills into evidence, and emits CostProfile-shaped values to feed back in.
#
# THE LOOP IT CLOSES:
#   live/shadow fill --> record() --> per-symbol aggregation --> updated
#   slippage_pct / spread_pct --> cost_adjusted_scoring uses measured costs -->
#   backtests reflect reality. (The learning_loop can pull these updates on its
#   schedule; this module just produces the evidence and the suggested values.)
#
# DESIGN PRINCIPLE (project-wide):
#   An estimate built on too few fills is not reported as a confident number.
#   observed_profile() returns per-field values only where the sample is large
#   enough, and flags the rest as "keep the existing assumption" -- measured
#   uncertainty must not masquerade as a precise correction. Slippage is also
#   recorded SIGNED (adverse vs favourable) so the aggregate cannot be flattered
#   by favourable fills cancelling adverse ones; the cost estimate uses adverse.
# ==============================================================================

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from discovery_config import DATA_DIR
    _DB_PATH = str(DATA_DIR / "slippage_observations.db")
except Exception:
    _D = Path(__file__).parent / "data"
    _D.mkdir(parents=True, exist_ok=True)
    _DB_PATH = str(_D / "slippage_observations.db")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Below this many fills, a per-symbol estimate is not considered reliable.
DEFAULT_MIN_FILLS = 30


@dataclass
class FillObservation:
    symbol: str
    side: str                 # 'buy' | 'sell'
    signal_price: float       # price when the signal fired / order was sent
    fill_price: float         # price actually filled at
    quoted_spread: float      # spread (price units) quoted at entry, if known
    timestamp: str = ""

    @property
    def slippage_pct(self) -> float:
        """
        Signed slippage as a percent of signal price, ADVERSE-POSITIVE.
        A buy filled ABOVE signal is adverse (positive). A sell filled BELOW
        signal is adverse (positive). Favourable fills are negative.
        """
        if self.signal_price == 0:
            return 0.0
        raw = (self.fill_price - self.signal_price) / self.signal_price * 100.0
        # For a sell, a lower fill is adverse, so flip the sign.
        if self.side in ("sell", "short"):
            raw = -raw
        return raw

    @property
    def spread_pct(self) -> float:
        if self.signal_price == 0 or self.quoted_spread <= 0:
            return 0.0
        return self.quoted_spread / self.signal_price * 100.0


@dataclass
class SymbolStats:
    symbol: str
    n_fills: int = 0
    mean_adverse_slippage_pct: float = 0.0   # mean of adverse (positive) slippage
    median_slippage_pct: float = 0.0         # signed median (sanity)
    p90_adverse_slippage_pct: float = 0.0
    mean_spread_pct: float = 0.0
    median_spread_pct: float = 0.0
    sufficient: bool = False
    note: str = ""


class SlippageRecorder:
    """Records fills and aggregates per-symbol slippage/spread observations."""

    def __init__(self, db_path: str = _DB_PATH,
                 min_fills: int = DEFAULT_MIN_FILLS):
        self.db_path = db_path
        self.min_fills = min_fills
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return conn

    def _ensure_tables(self) -> None:
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                side          TEXT NOT NULL,
                signal_price  REAL NOT NULL,
                fill_price    REAL NOT NULL,
                quoted_spread REAL DEFAULT 0,
                slippage_pct  REAL NOT NULL,
                spread_pct    REAL DEFAULT 0,
                timestamp     TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_symbol "
                     "ON fills(symbol)")
        conn.commit()
        conn.close()

    # -- Recording -------------------------------------------------------------
    def record(self, symbol: str, side: str, signal_price: float,
               fill_price: float, quoted_spread: float = 0.0,
               timestamp: Optional[str] = None) -> FillObservation:
        """Record one fill. Returns the observation (with computed slippage)."""
        obs = FillObservation(
            symbol=symbol, side=str(side).lower(),
            signal_price=float(signal_price), fill_price=float(fill_price),
            quoted_spread=float(quoted_spread),
            timestamp=timestamp or _utcnow())
        conn = self._conn()
        conn.execute(
            "INSERT INTO fills (symbol, side, signal_price, fill_price, "
            "quoted_spread, slippage_pct, spread_pct, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (obs.symbol, obs.side, obs.signal_price, obs.fill_price,
             obs.quoted_spread, obs.slippage_pct, obs.spread_pct, obs.timestamp))
        conn.commit()
        conn.close()
        return obs

    def record_observation(self, obs: FillObservation) -> None:
        self.record(obs.symbol, obs.side, obs.signal_price, obs.fill_price,
                    obs.quoted_spread, obs.timestamp)

    # -- Aggregation -----------------------------------------------------------
    def stats_for(self, symbol: str) -> SymbolStats:
        conn = self._conn()
        rows = conn.execute(
            "SELECT slippage_pct, spread_pct FROM fills WHERE symbol = ?",
            (symbol,)).fetchall()
        conn.close()

        n = len(rows)
        if n == 0:
            return SymbolStats(symbol=symbol, n_fills=0, sufficient=False,
                               note="no fills recorded")

        slippages = [r["slippage_pct"] for r in rows]
        spreads = [r["spread_pct"] for r in rows if r["spread_pct"] > 0]

        # Adverse slippage only (positive values); favourable fills excluded
        # from the cost estimate so they cannot flatter it.
        adverse = [s for s in slippages if s > 0]

        stats = SymbolStats(symbol=symbol, n_fills=n)
        stats.median_slippage_pct = statistics.median(slippages)
        if adverse:
            stats.mean_adverse_slippage_pct = statistics.mean(adverse)
            stats.p90_adverse_slippage_pct = _percentile(sorted(adverse), 90)
        if spreads:
            stats.mean_spread_pct = statistics.mean(spreads)
            stats.median_spread_pct = statistics.median(spreads)

        stats.sufficient = n >= self.min_fills
        if not stats.sufficient:
            stats.note = (f"only {n} fills (< {self.min_fills}); estimate "
                          f"not yet reliable")
        return stats

    def all_symbols(self) -> List[str]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM fills ORDER BY symbol").fetchall()
        conn.close()
        return [r["symbol"] for r in rows]

    # -- Feedback to the cost model -------------------------------------------
    def observed_profile(self, symbol: str,
                         base_profile: Any = None) -> Dict[str, Any]:
        """
        Produce cost-model updates for a symbol from observed fills.

        Returns a dict with suggested slippage_pct and spread_pct plus metadata.
        Where the sample is insufficient, the corresponding field is None and
        the caller should KEEP its existing assumption -- this never overwrites a
        deliberate assumption with a noisy measurement.

        If base_profile (a CostProfile-like object) is given, its values are
        echoed as the fallback for insufficient fields.
        """
        stats = self.stats_for(symbol)
        base_slip = getattr(base_profile, "slippage_pct", None)
        base_spread = getattr(base_profile, "spread_pct", None)

        out: Dict[str, Any] = {
            "symbol": symbol,
            "n_fills": stats.n_fills,
            "sufficient": stats.sufficient,
            "note": stats.note,
            "slippage_pct": None,
            "spread_pct": None,
            "fallback_slippage_pct": base_slip,
            "fallback_spread_pct": base_spread,
        }
        if stats.sufficient:
            # Use mean adverse slippage as the cost estimate (conservative vs
            # median, which would understate the tail that actually hurts).
            out["slippage_pct"] = round(stats.mean_adverse_slippage_pct, 6)
            if stats.mean_spread_pct > 0:
                out["spread_pct"] = round(stats.mean_spread_pct, 6)
        return out

    def clear(self, symbol: Optional[str] = None) -> None:
        conn = self._conn()
        if symbol is None:
            conn.execute("DELETE FROM fills")
        else:
            conn.execute("DELETE FROM fills WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


__all__ = ["SlippageRecorder", "FillObservation", "SymbolStats"]


if __name__ == "__main__":
    import tempfile, os, random
    db = tempfile.mktemp(suffix=".db")
    rec = SlippageRecorder(db_path=db, min_fills=30)
    random.seed(0)
    for _ in range(50):
        sig = 1.10
        # buys fill slightly above signal (adverse), ~0.3 pip mean
        fill = sig + random.uniform(-0.00002, 0.00006)
        rec.record("EURUSD", "buy", sig, fill, quoted_spread=0.00002)
    prof = rec.observed_profile("EURUSD")
    print("EURUSD fills:", prof["n_fills"], "sufficient:", prof["sufficient"])
    print("observed slippage_pct:", prof["slippage_pct"])
    print("observed spread_pct:", prof["spread_pct"])
    os.remove(db)
