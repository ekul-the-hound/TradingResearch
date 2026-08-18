# ==============================================================================
# weekend_policy.py -- Weekend / End-of-Day Flatten Policy + Monday-Gap Stress
# ==============================================================================
# Two related jobs:
#
#   1. FLATTEN TIMING. Decide when to auto-close positions before a boundary:
#        * before the Friday market close (avoid holding over the weekend), and
#        * before Prague midnight (keep positions intraday so the FTMO daily-loss
#          anchor stays exact -- a position that never crosses midnight has no
#          overnight floating-P&L ambiguity in the daily calculation).
#
#   2. MONDAY-GAP STRESS. Measure how far price gapped over historical weekends
#      (Friday close -> Monday/Sunday-evening open) per symbol, so a strategy
#      that holds over weekends can be stressed against realistic gap risk.
#
# IT REUSES THE PROJECT'S DST-CORRECT PRAGUE LOGIC:
#   Boundary timing calls ftmo_daily_anchor.prague_date_of / prague_midnight_utc
#   rather than re-deriving Prague midnight (which is 23:00 UTC in winter, 22:00
#   in summer). Reimplementing that is exactly how a daily-loss boundary drifts
#   an hour half the year. If those helpers are unavailable, the flatten policy
#   degrades to a clearly-labelled UTC approximation instead of guessing.
#
# DESIGN PRINCIPLE (project-wide):
#   The safe default near a boundary is to FLATTEN. If the module cannot tell
#   how close midnight is (missing tz support, unparseable time), it says so and
#   recommends flattening rather than silently assuming there is plenty of time.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

# Reuse the canonical, DST-correct Prague helpers when available.
try:
    from ftmo_daily_anchor import prague_date_of, prague_midnight_utc
    _HAVE_PRAGUE = True
except Exception:
    _HAVE_PRAGUE = False


FRIDAY = 4   # datetime.weekday(): Monday=0 ... Sunday=6


@dataclass
class WeekendPolicyConfig:
    # Flatten this many minutes before the Friday close.
    flatten_before_friday_close_minutes: int = 30
    # The Friday close time, in the exchange/broker local clock, "HH:MM".
    # FX typically closes ~21:00-22:00 UTC Friday; set to your broker's value.
    friday_close_hhmm_utc: str = "21:00"
    # Flatten this many minutes before Prague midnight (intraday enforcement).
    flatten_before_midnight_minutes: int = 5
    # Whether each rule is active.
    enforce_friday_close: bool = True
    enforce_prague_midnight: bool = True


@dataclass
class FlattenDecision:
    should_flatten: bool
    reason: str = ""
    minutes_to_boundary: Optional[float] = None
    boundary: str = ""   # 'friday_close' | 'prague_midnight' | ''

    def __bool__(self) -> bool:
        return self.should_flatten


class WeekendPolicy:
    """Decides when to flatten before weekend / daily boundaries."""

    def __init__(self, config: Optional[WeekendPolicyConfig] = None):
        self.config = config or WeekendPolicyConfig()

    def check(self, now: Optional[datetime] = None) -> FlattenDecision:
        """
        Decide whether to flatten right now. `now` should be timezone-aware UTC;
        a naive datetime is assumed to be UTC.
        """
        cfg = self.config
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Evaluate both boundaries; the nearer triggering one wins.
        decisions: List[FlattenDecision] = []

        if cfg.enforce_prague_midnight:
            decisions.append(self._check_prague_midnight(now))
        if cfg.enforce_friday_close:
            decisions.append(self._check_friday_close(now))

        triggering = [d for d in decisions if d.should_flatten]
        if triggering:
            # Choose the one with the least time to its boundary.
            triggering.sort(key=lambda d: (d.minutes_to_boundary
                                           if d.minutes_to_boundary is not None
                                           else 1e9))
            return triggering[0]

        # None triggered: report the nearest boundary for visibility.
        informative = [d for d in decisions
                       if d.minutes_to_boundary is not None]
        if informative:
            informative.sort(key=lambda d: d.minutes_to_boundary or 0.0)
            near = informative[0]
            return FlattenDecision(
                False, f"within limits; {near.minutes_to_boundary:.0f} min to "
                       f"{near.boundary}", near.minutes_to_boundary, near.boundary)
        return FlattenDecision(False, "no boundary rules active")

    # -- Prague midnight -------------------------------------------------------
    def _check_prague_midnight(self, now: datetime) -> FlattenDecision:
        cfg = self.config
        if not _HAVE_PRAGUE:
            # Degrade honestly: we cannot compute DST-correct Prague midnight.
            return FlattenDecision(
                True, "Prague tz support unavailable; flattening as the safe "
                      "default rather than guessing the midnight boundary",
                None, "prague_midnight")
        try:
            naive_utc = now.astimezone(timezone.utc).replace(tzinfo=None)
            today_prague = prague_date_of(naive_utc)
            # Next Prague midnight = midnight of the following Prague date.
            next_mid_naive = prague_midnight_utc(today_prague + timedelta(days=1))
            next_mid = next_mid_naive.replace(tzinfo=timezone.utc)
            minutes = (next_mid - now).total_seconds() / 60.0
        except Exception as e:
            return FlattenDecision(
                True, f"could not compute Prague midnight ({e}); flattening as "
                      f"the safe default", None, "prague_midnight")

        if 0 <= minutes <= cfg.flatten_before_midnight_minutes:
            return FlattenDecision(
                True, f"{minutes:.1f} min to Prague midnight "
                      f"(<= {cfg.flatten_before_midnight_minutes} min buffer)",
                minutes, "prague_midnight")
        return FlattenDecision(False, "", minutes, "prague_midnight")

    # -- Friday close ----------------------------------------------------------
    def _check_friday_close(self, now: datetime) -> FlattenDecision:
        cfg = self.config
        close = _parse_hhmm(cfg.friday_close_hhmm_utc)
        if close is None:
            return FlattenDecision(False, "", None, "friday_close")

        # Build this week's Friday close instant in UTC.
        close_h, close_m = close
        # Find the Friday of the current UTC week.
        days_to_friday = (FRIDAY - now.weekday())
        friday_date = (now + timedelta(days=days_to_friday)).date()
        friday_close = datetime(friday_date.year, friday_date.month,
                                friday_date.day, close_h, close_m,
                                tzinfo=timezone.utc)
        minutes = (friday_close - now).total_seconds() / 60.0

        # Only relevant as we approach Friday close from before it.
        if 0 <= minutes <= cfg.flatten_before_friday_close_minutes:
            return FlattenDecision(
                True, f"{minutes:.1f} min to Friday close "
                      f"(<= {cfg.flatten_before_friday_close_minutes} min buffer)",
                minutes, "friday_close")
        # If we're past this week's Friday close but before Sunday reopen, the
        # market is shut -- holding is moot, but flag it.
        if now.weekday() == FRIDAY and minutes < 0:
            return FlattenDecision(
                False, "past Friday close", None, "friday_close")
        return FlattenDecision(False, "",
                               minutes if minutes >= 0 else None, "friday_close")


# ==============================================================================
# MONDAY-GAP STRESS
# ==============================================================================
@dataclass
class GapStats:
    n_weekends: int = 0
    mean_gap_pct: float = 0.0
    median_gap_pct: float = 0.0
    p95_gap_pct: float = 0.0
    max_gap_pct: float = 0.0
    worst_gap_pct: float = 0.0   # signed, most negative
    sufficient: bool = True
    note: str = ""


def weekend_gaps(bars: Any, min_weekends: int = 10) -> GapStats:
    """
    Measure weekend gaps from a bar DataFrame with a datetime index and a
    'close'/'open' column. A "weekend gap" is the move from the last Friday-side
    close to the next session's first open across a >1-day calendar jump.

    Returns absolute-gap distribution plus the worst signed gap. Reports
    insufficient rather than a confident number when too few weekends exist.
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        return GapStats(sufficient=False, note=f"pandas/numpy unavailable: {e}")

    try:
        if bars is None or len(bars) < 2:
            return GapStats(sufficient=False, note="need at least 2 bars")
        df = bars.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try a 'date'/'timestamp' column.
            for c in ("date", "timestamp", "time"):
                if c in df.columns:
                    df.index = pd.to_datetime(df[c])
                    break
        idx = pd.to_datetime(df.index)
        close = df["close"].to_numpy(dtype=float)
        open_ = (df["open"].to_numpy(dtype=float)
                 if "open" in df.columns else close)

        gaps_pct: List[float] = []
        for i in range(1, len(df)):
            delta_days = (idx[i] - idx[i - 1]).days
            if delta_days >= 2:  # a weekend / multi-day jump
                prev_close = close[i - 1]
                nxt_open = open_[i]
                if prev_close:
                    gaps_pct.append((nxt_open - prev_close) / prev_close * 100.0)

        if len(gaps_pct) < min_weekends:
            return GapStats(
                n_weekends=len(gaps_pct), sufficient=False,
                note=f"only {len(gaps_pct)} weekend gaps "
                     f"(< {min_weekends}); distribution unreliable")

        arr = np.array(gaps_pct)
        absa = np.abs(arr)
        return GapStats(
            n_weekends=len(gaps_pct),
            mean_gap_pct=float(absa.mean()),
            median_gap_pct=float(np.median(absa)),
            p95_gap_pct=float(np.percentile(absa, 95)),
            max_gap_pct=float(absa.max()),
            worst_gap_pct=float(arr.min()),
            sufficient=True,
        )
    except Exception as e:
        return GapStats(sufficient=False, note=f"gap computation failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_hhmm(s: str):
    try:
        h, m = s.strip().split(":")
        h, m = int(h), int(m)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return (h, m)
    except (ValueError, AttributeError):
        pass
    return None


__all__ = ["WeekendPolicy", "WeekendPolicyConfig", "FlattenDecision",
           "weekend_gaps", "GapStats"]


if __name__ == "__main__":
    pol = WeekendPolicy()
    # A Wednesday midday -> no flatten.
    wed = datetime(2026, 1, 7, 12, 0, tzinfo=timezone.utc)
    print("wed midday:", pol.check(wed).reason)
    # Just before Prague midnight (23:58 UTC winter ~ 00:58 Prague next day is
    # wrong; winter midnight Prague = 23:00 UTC, so 22:57 UTC is 3 min before).
    near = datetime(2026, 1, 7, 22, 57, tzinfo=timezone.utc)
    d = pol.check(near)
    print("near midnight:", d.should_flatten, d.reason)