# ==============================================================================
# parity_harness.py -- Backtest <-> Live Trade Parity Checker
# ==============================================================================
# Replays the SAME data through the backtest path and the live path, then proves
# they produce the SAME trades. This is the safety net for the failure mode where
# a strategy passes every backtest gate and then behaves differently live because
# a fill, a signal, or a sizing calc silently diverged between the two engines.
# It is also a hard prerequisite for ever trusting a vectorized fast-path engine.
#
# TWO PARTS:
#   1. ParityChecker (this file, fully tested) -- the comparison ENGINE. Given
#      two trade lists it decides whether they match, distinguishing:
#        * HARD mismatches that must never differ: trade count, ordering,
#          direction (long/short), and size. A difference here means the two
#          engines disagree about WHAT they traded -- a real bug.
#        * SOFT differences that may legitimately differ within tolerance:
#          fill prices and timestamps (slippage models, clock granularity).
#          These are reported and bounded, not treated as automatic failures.
#
#   2. run_parity() scaffold -- wires your REAL backtester and live engine in.
#      Running it needs Backtrader and a live-engine instance with a replay feed,
#      so it is meant to be executed in your environment, not here. The checker
#      it delegates to is what carries the correctness, and that is fully tested.
#
# DESIGN PRINCIPLE (project-wide):
#   A "pass" must mean the engines genuinely agree, not that the check was lax.
#   Tolerances are explicit and narrow; anything outside them is surfaced. When
#   the two lists cannot even be aligned (different counts), that is a hard fail
#   reported loudly rather than smoothed over by truncating to the shorter list.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


# ── Normalized trade record ───────────────────────────────────────────────────
@dataclass
class Trade:
    """
    A single round-trip trade, normalized so backtest and live records compare
    on equal footing. Both engines map their native records into this.
    """
    direction: str                 # 'long' | 'short'
    size: float
    entry_price: float
    exit_price: float
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None

    @staticmethod
    def from_backtest(rec: Dict[str, Any]) -> "Trade":
        """Map a backtester_multi_timeframe trade_record dict to a Trade."""
        is_long = rec.get("is_long")
        if is_long is None:
            is_long = float(rec.get("size", 0)) >= 0
        return Trade(
            direction="long" if is_long else "short",
            size=abs(float(rec.get("size", 0) or 0)),
            entry_price=float(rec.get("entry_price", 0) or 0),
            exit_price=float(rec.get("exit_price", 0) or 0),
            entry_time=_to_iso(rec.get("entry_date")),
            exit_time=_to_iso(rec.get("exit_date")),
            pnl=_maybe_float(rec.get("pnl")),
        )

    @staticmethod
    def from_live(rec: Dict[str, Any]) -> "Trade":
        """
        Map a live/shadow trade dict to a Trade. Live records vary; this reads
        the common keys and tolerates missing timestamps.
        """
        direction = rec.get("direction")
        if direction is None:
            side = str(rec.get("side", "")).lower()
            if side in ("buy", "long"):
                direction = "long"
            elif side in ("sell", "short"):
                direction = "short"
            else:
                direction = "long" if float(rec.get("size", 0) or 0) >= 0 else "short"
        return Trade(
            direction=direction,
            size=abs(float(rec.get("size", 0) or 0)),
            entry_price=float(rec.get("entry_price", rec.get("entry", 0)) or 0),
            exit_price=float(rec.get("exit_price", rec.get("exit", 0)) or 0),
            entry_time=_to_iso(rec.get("entry_time", rec.get("entry_date"))),
            exit_time=_to_iso(rec.get("exit_time", rec.get("exit_date"))),
            pnl=_maybe_float(rec.get("pnl")),
        )


# ── Tolerances / config ───────────────────────────────────────────────────────
@dataclass
class ParityConfig:
    # Absolute price tolerance for fills (in price units). Fills may differ by a
    # slippage model, but not by much. Default: 1 pip on a 5-dp FX pair.
    price_tolerance: float = 1e-4
    # Relative size tolerance (fraction). Size is a HARD field; keep this tiny --
    # only floating-point noise should slip through.
    size_rel_tolerance: float = 1e-6
    # Max allowed timestamp difference in seconds (0 disables timestamp checks).
    time_tolerance_seconds: float = 60.0
    # If True, pnl is compared (within price_tolerance * size); many setups only
    # care about trade structure, so this is off by default.
    compare_pnl: bool = False


# ── Per-trade + overall results ───────────────────────────────────────────────
@dataclass
class TradeDiff:
    index: int
    hard_mismatch: bool
    fields: List[str] = field(default_factory=list)  # which fields differed
    detail: str = ""


@dataclass
class ParityResult:
    matched: bool
    n_backtest: int
    n_live: int
    hard_mismatches: List[TradeDiff] = field(default_factory=list)
    soft_diffs: List[TradeDiff] = field(default_factory=list)
    summary: str = ""

    def __bool__(self) -> bool:
        return self.matched


# ── The checker ───────────────────────────────────────────────────────────────
class ParityChecker:
    """Compares two normalized trade lists under explicit tolerances."""

    def __init__(self, config: Optional[ParityConfig] = None):
        self.config = config or ParityConfig()

    def compare(self, backtest: Sequence[Trade],
                live: Sequence[Trade]) -> ParityResult:
        cfg = self.config
        nb, nl = len(backtest), len(live)

        # 1. Count mismatch is a HARD fail. Do not truncate to align -- a
        #    different number of trades means the engines fundamentally disagree.
        if nb != nl:
            return ParityResult(
                matched=False, n_backtest=nb, n_live=nl,
                hard_mismatches=[TradeDiff(
                    index=-1, hard_mismatch=True, fields=["count"],
                    detail=f"trade count differs: backtest={nb} live={nl}")],
                summary=f"HARD FAIL: {nb} backtest trades vs {nl} live trades",
            )

        hard: List[TradeDiff] = []
        soft: List[TradeDiff] = []

        for i, (b, l) in enumerate(zip(backtest, live)):
            hard_fields: List[str] = []
            soft_fields: List[str] = []

            # -- HARD: direction must be identical. ---------------------------
            if b.direction != l.direction:
                hard_fields.append("direction")

            # -- HARD: size must match within floating-point noise. -----------
            if not _rel_close(b.size, l.size, cfg.size_rel_tolerance):
                hard_fields.append("size")

            # -- SOFT: entry/exit fills within price tolerance. ---------------
            if not _abs_close(b.entry_price, l.entry_price, cfg.price_tolerance):
                soft_fields.append("entry_price")
            if not _abs_close(b.exit_price, l.exit_price, cfg.price_tolerance):
                soft_fields.append("exit_price")

            # -- SOFT: timestamps within tolerance (if both present). ---------
            if cfg.time_tolerance_seconds > 0:
                for tname, bt_t, lv_t in (("entry_time", b.entry_time, l.entry_time),
                                          ("exit_time", b.exit_time, l.exit_time)):
                    dt = _time_gap_seconds(bt_t, lv_t)
                    if dt is not None and dt > cfg.time_tolerance_seconds:
                        soft_fields.append(tname)

            # -- Optional: pnl. -----------------------------------------------
            if cfg.compare_pnl and b.pnl is not None and l.pnl is not None:
                pnl_tol = cfg.price_tolerance * max(b.size, 1.0)
                if not _abs_close(b.pnl, l.pnl, pnl_tol):
                    soft_fields.append("pnl")

            if hard_fields:
                hard.append(TradeDiff(
                    index=i, hard_mismatch=True, fields=hard_fields,
                    detail=_diff_detail(i, b, l, hard_fields)))
            if soft_fields:
                soft.append(TradeDiff(
                    index=i, hard_mismatch=False, fields=soft_fields,
                    detail=_diff_detail(i, b, l, soft_fields)))

        matched = len(hard) == 0
        if matched and not soft:
            summary = f"PARITY OK: {nb} trades match exactly (within tolerance)"
        elif matched:
            summary = (f"PARITY OK with {len(soft)} within-tolerance difference(s) "
                       f"across {nb} trades")
        else:
            summary = (f"PARITY FAIL: {len(hard)} hard mismatch(es) "
                       f"across {nb} trades")

        return ParityResult(
            matched=matched, n_backtest=nb, n_live=nl,
            hard_mismatches=hard, soft_diffs=soft, summary=summary,
        )

    def compare_raw(self, backtest_records: Sequence[Dict[str, Any]],
                    live_records: Sequence[Dict[str, Any]]) -> ParityResult:
        """Convenience: map raw engine dicts to Trades, then compare."""
        bt = [Trade.from_backtest(r) for r in backtest_records]
        lv = [Trade.from_live(r) for r in live_records]
        return self.compare(bt, lv)


# ── run scaffold (wire your real engines here; run locally) ───────────────────
def run_parity(strategy_path: str, data,
               config: Optional[ParityConfig] = None) -> ParityResult:
    """
    Replay one strategy through both engines on the same data and compare.

    This is a SCAFFOLD: the two _run_* helpers below must be completed with your
    actual backtester and live-engine calls. They are separated so the wiring is
    obvious and the pure comparison above stays fully tested. Running this needs
    Backtrader + a live-engine replay feed, so execute it in your environment.
    """
    backtest_records = _run_backtest(strategy_path, data)
    live_records = _run_live_replay(strategy_path, data)
    return ParityChecker(config).compare_raw(backtest_records, live_records)


def _run_backtest(strategy_path: str, data) -> List[Dict[str, Any]]:
    """
    TODO(local): run the strategy through backtester_multi_timeframe and return
    its trade_record dicts (get_analysis()['trades']). Left unimplemented here
    because Backtrader is not available in the build sandbox.
    """
    raise NotImplementedError(
        "wire backtester_multi_timeframe here and return trade_record dicts")


def _run_live_replay(strategy_path: str, data) -> List[Dict[str, Any]]:
    """
    TODO(local): drive live_engine with a PaperBroker fed the SAME bars as the
    backtest (bar-by-bar), then return the shadow trader's trade dicts. Left
    unimplemented here because it needs a live-engine instance + replay feed.
    """
    raise NotImplementedError(
        "wire live_engine + PaperBroker replay here and return trade dicts")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _abs_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _rel_close(a: float, b: float, rel: float) -> bool:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale <= rel


def _maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    txt = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(txt)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
    return None


def _time_gap_seconds(a: Optional[str], b: Optional[str]) -> Optional[float]:
    da, db = _parse_dt(a), _parse_dt(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds())


def _diff_detail(i: int, b: Trade, l: Trade, fields: List[str]) -> str:
    bits = []
    for f in fields:
        if f == "count":
            continue
        bv = getattr(b, f, None)
        lv = getattr(l, f, None)
        bits.append(f"{f}: backtest={bv} live={lv}")
    return f"trade[{i}] " + "; ".join(bits)


__all__ = [
    "Trade", "ParityConfig", "ParityChecker", "ParityResult", "TradeDiff",
    "run_parity",
]


if __name__ == "__main__":
    # Self-demo of the checker (the part that runs anywhere).
    checker = ParityChecker()
    bt = [Trade("long", 1.0, 1.1000, 1.1050),
          Trade("short", 2.0, 1.2000, 1.1950)]
    lv = [Trade("long", 1.0, 1.10001, 1.10499),   # fills off by <1 pip
          Trade("short", 2.0, 1.2000, 1.1950)]
    res = checker.compare(bt, lv)
    print(res.summary)
    for d in res.soft_diffs:
        print("  soft:", d.detail)
