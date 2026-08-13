# ==============================================================================
# live_tick_guard.py -- Live Bad-Tick Filter + Max-Spread Guard
# ==============================================================================
# A pre-trade sanity layer for live/shadow trading. It inspects each incoming
# price tick and decides whether the market is currently CLEAN enough to act on.
#
# WHY THIS EXISTS:
#   Live feeds produce garbage: frozen quotes when a feed stalls, crossed or
#   locked bid/ask, zero/negative prices, and single-print spikes N sigma off
#   the recent mid. Acting on any of these can open a position at a fictional
#   price or trip a daily-loss rule on a print that never really existed.
#
#   The guard is deliberately CONSERVATIVE about calling something bad, because
#   a false positive that rejects a real fast move is far cheaper in a prop
#   challenge than a false negative that fills on a bad print. But it is tunable
#   so it does not eat legitimate volatility (NFP, CPI) wholesale.
#
# WHAT IT IS NOT:
#   Not a strategy signal. Not a fill simulator. It answers exactly one question
#   per tick: "is this quote trustworthy enough to enter/exit on right now?"
#
# USAGE:
#   from live_tick_guard import LiveTickGuard, TickGuardConfig
#   guard = LiveTickGuard(TickGuardConfig(max_spread_bps=8.0))
#   verdict = guard.check(tick)          # tick is a broker_base.BrokerTick
#   if verdict.ok:
#       ... proceed to place/close order ...
#   else:
#       ... log verdict.reasons, skip this tick ...
#
# The guard keeps a short rolling history of accepted mids PER SYMBOL so it can
# detect staleness (unchanged price for too long) and outlier prints. History
# is bounded and in-memory only.
# ==============================================================================

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Any


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class TickGuardConfig:
    """
    Thresholds for the tick guard. Defaults are deliberately conservative for
    liquid FX majors; widen them per-symbol for less liquid instruments.
    """
    # Spread: reject quotes wider than this many basis points of mid.
    max_spread_bps: float = 10.0

    # Staleness: reject if the mid has not moved for this many consecutive
    # ticks (a frozen feed). 0 disables the check.
    max_frozen_ticks: int = 20

    # Outlier: reject if a mid is more than this many robust-sigma away from the
    # recent median mid. Uses MAD (median absolute deviation), which is not
    # dragged around by the very spike it is trying to catch. 0 disables.
    outlier_sigma: float = 8.0

    # How many recent accepted mids to keep per symbol for the outlier/stale
    # checks. Must be >= min_history_for_outlier to ever fire the outlier rule.
    history_len: int = 64

    # Minimum accepted-tick history before the outlier check is allowed to fire.
    # Below this we do not have a stable enough median to judge outliers.
    min_history_for_outlier: int = 20

    # If True, a tick that fails staleness/outlier is NOT added to history, so a
    # burst of bad ticks cannot poison the baseline. Recommended True.
    exclude_rejected_from_history: bool = True


# ── Verdict ───────────────────────────────────────────────────────────────────
@dataclass
class TickVerdict:
    """Result of checking one tick."""
    ok: bool
    symbol: str
    mid: float
    spread_bps: float
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


# ── The guard ─────────────────────────────────────────────────────────────────
class LiveTickGuard:
    """
    Stateful, per-symbol tick sanity checker.

    Thread-safety: not thread-safe. Use one guard per feed-consuming thread, or
    wrap check() in a lock if you share it.
    """

    def __init__(self, config: Optional[TickGuardConfig] = None):
        self.config = config or TickGuardConfig()
        # Per-symbol rolling history of accepted mids.
        self._history: Dict[str, Deque[float]] = {}
        # Per-symbol count of consecutive identical mids (staleness tracking).
        self._frozen_count: Dict[str, int] = {}
        self._last_mid: Dict[str, float] = {}
        # Rejection tally for observability.
        self.rejections: Dict[str, int] = {}

    # -- Public API ------------------------------------------------------------
    def check(self, tick: Any) -> TickVerdict:
        """
        Inspect one tick. `tick` must expose .symbol, .bid, .ask and (ideally)
        the .mid / .spread_bps properties of broker_base.BrokerTick. We read
        bid/ask directly rather than trusting a precomputed mid, so a broker
        that fabricates .mid cannot slip a bad quote past us.
        """
        symbol = getattr(tick, "symbol", "") or ""
        bid = _as_float(getattr(tick, "bid", None))
        ask = _as_float(getattr(tick, "ask", None))
        last = _as_float(getattr(tick, "last", None))

        reasons: List[str] = []

        # 1. Structural validity of the quote itself. -------------------------
        # A price of exactly 0.0, negative, NaN or inf is never a real quote.
        bid_ok = _is_positive_finite(bid)
        ask_ok = _is_positive_finite(ask)

        if not bid_ok and not ask_ok:
            # No usable two-sided quote. Fall back to `last` only to report a
            # mid for logging; still reject.
            mid = last if _is_positive_finite(last) else 0.0
            reasons.append("no valid bid or ask (both zero/negative/NaN)")
            return self._reject(symbol, mid, 0.0, reasons)

        if not bid_ok:
            reasons.append("invalid bid (zero/negative/NaN)")
        if not ask_ok:
            reasons.append("invalid ask (zero/negative/NaN)")

        # If exactly one side is bad we cannot form a trustworthy mid/spread.
        if reasons:
            mid = ask if ask_ok else bid
            return self._reject(symbol, mid, 0.0, reasons)

        # 2. Crossed or locked market. ----------------------------------------
        # Crossed: bid > ask (impossible in a sane book). Locked: bid == ask
        # (zero spread) is almost always a feed artifact in FX, not a real market.
        if bid > ask:
            mid = (bid + ask) / 2.0
            reasons.append(f"crossed market: bid {bid} > ask {ask}")
            return self._reject(symbol, mid, 0.0, reasons)
        if bid == ask:
            reasons.append(f"locked market: bid == ask == {bid} (zero spread)")
            return self._reject(symbol, bid, 0.0, reasons)

        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000.0

        # 3. Spread guard. -----------------------------------------------------
        if self.config.max_spread_bps > 0 and spread_bps > self.config.max_spread_bps:
            reasons.append(
                f"spread {spread_bps:.2f} bps exceeds max "
                f"{self.config.max_spread_bps:.2f} bps"
            )
            return self._reject(symbol, mid, spread_bps, reasons)

        # 4. Staleness (frozen feed). -----------------------------------------
        if self.config.max_frozen_ticks > 0:
            prev = self._last_mid.get(symbol)
            if prev is not None and mid == prev:
                self._frozen_count[symbol] = self._frozen_count.get(symbol, 0) + 1
            else:
                self._frozen_count[symbol] = 0
            if self._frozen_count[symbol] >= self.config.max_frozen_ticks:
                reasons.append(
                    f"stale feed: mid unchanged for "
                    f"{self._frozen_count[symbol]} ticks"
                )
                # Do not update _last_mid here; keep counting until it moves.
                return self._reject(symbol, mid, spread_bps, reasons)

        # 5. Outlier print (robust). ------------------------------------------
        hist = self._history.get(symbol)
        if (self.config.outlier_sigma > 0 and hist is not None
                and len(hist) >= self.config.min_history_for_outlier):
            med = _median(hist)
            mad = _median([abs(x - med) for x in hist])
            # 1.4826 scales MAD to be a consistent estimator of sigma for
            # normally-distributed data.
            robust_sigma = 1.4826 * mad
            if robust_sigma > 0:
                deviation = abs(mid - med) / robust_sigma
                if deviation > self.config.outlier_sigma:
                    reasons.append(
                        f"outlier print: mid {mid} is {deviation:.1f} robust-sigma "
                        f"from recent median {med} (limit {self.config.outlier_sigma})"
                    )
                    return self._reject(symbol, mid, spread_bps, reasons)

        # ACCEPTED. Update state. ---------------------------------------------
        self._last_mid[symbol] = mid
        self._push_history(symbol, mid)
        return TickVerdict(ok=True, symbol=symbol, mid=mid,
                           spread_bps=spread_bps, reasons=[])

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear history for one symbol, or all symbols if None."""
        if symbol is None:
            self._history.clear()
            self._frozen_count.clear()
            self._last_mid.clear()
            self.rejections.clear()
        else:
            self._history.pop(symbol, None)
            self._frozen_count.pop(symbol, None)
            self._last_mid.pop(symbol, None)
            self.rejections.pop(symbol, None)

    def stats(self) -> Dict[str, Any]:
        """Observability snapshot for logging/dashboards."""
        return {
            "symbols_tracked": sorted(self._history.keys()),
            "history_sizes": {s: len(h) for s, h in self._history.items()},
            "frozen_counts": dict(self._frozen_count),
            "rejections": dict(self.rejections),
        }

    # -- Internals -------------------------------------------------------------
    def _reject(self, symbol: str, mid: float, spread_bps: float,
                reasons: List[str]) -> TickVerdict:
        self.rejections[symbol] = self.rejections.get(symbol, 0) + 1
        if not self.config.exclude_rejected_from_history:
            self._push_history(symbol, mid)
        return TickVerdict(ok=False, symbol=symbol, mid=mid,
                           spread_bps=spread_bps, reasons=reasons)

    def _push_history(self, symbol: str, mid: float) -> None:
        h = self._history.get(symbol)
        if h is None:
            h = deque(maxlen=self.config.history_len)
            self._history[symbol] = h
        h.append(mid)


# ── Small dependency-free helpers ─────────────────────────────────────────────
def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_positive_finite(v: Optional[float]) -> bool:
    return v is not None and math.isfinite(v) and v > 0.0


def _median(values) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


if __name__ == "__main__":
    # Tiny self-demo when run directly (not a substitute for the test file).
    from dataclasses import dataclass as _dc

    @_dc
    class _T:
        symbol: str
        bid: float
        ask: float
        last: float = 0.0

    g = LiveTickGuard(TickGuardConfig(max_spread_bps=10.0, max_frozen_ticks=3,
                                      outlier_sigma=8.0,
                                      min_history_for_outlier=5))
    samples = [
        _T("EURUSD", 1.08000, 1.08002),   # clean
        _T("EURUSD", 1.08010, 1.08005),   # crossed
        _T("EURUSD", 1.08000, 1.08000),   # locked
        _T("EURUSD", 1.08000, 1.08050),   # wide spread
        _T("EURUSD", 0.0, 1.08000),       # bad bid
    ]
    for t in samples:
        v = g.check(t)
        print(f"{t.symbol} bid={t.bid} ask={t.ask} -> "
              f"{'OK' if v.ok else 'REJECT'} {v.reasons}")
