# ==============================================================================
# mfe_mae_analyzer.py -- Stop/Target Placement Analytics from MFE/MAE
# ==============================================================================
# Turns per-trade excursion data into ACTIONABLE stop and target guidance.
#
# IMPORTANT -- THIS DOES NOT RECOMPUTE MFE/MAE:
#   The per-trade Maximum Favorable / Adverse Excursion is already computed by
#   intrabar_risk.trade_excursions(), which walks each trade's intrabar price
#   path. This module CONSUMES those TradeExcursion records (or equivalent
#   dicts) and aggregates them into placement statistics. Reusing the existing,
#   validated computation avoids two divergent MFE/MAE implementations.
#
# THE QUESTIONS IT ANSWERS:
#   * How tight could the stop have been without cutting winners short? For each
#     WINNING trade, how far did it go against you (MAE) before working out? The
#     distribution of winners' MAE tells you the minimum stop that keeps your
#     winners. A stop tighter than most winners' MAE is cutting good trades.
#   * How much did fixed targets leave on the table? For each trade, MFE minus
#     the realised move is unrealised run that a target could have captured (or,
#     for winners, that a trailing exit missed). Aggregated, this flags targets
#     set too close.
#   * Are stops routinely too wide? If losers' MAE greatly exceeds the realised
#     loss rarely, stops may be wider than necessary, wasting daily-loss budget.
#
# DESIGN PRINCIPLE (project-wide):
#   Guidance is described as evidence from the sample, never as a promise. The
#   output reports distributions ("75% of winners drew down no more than X")
#   and is explicit that these are in-sample observations to validate OOS, not
#   optimal parameters to fit. It refuses to emit guidance from too few trades
#   rather than presenting a confident number built on noise.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Minimum sample sizes below which a statistic is not reported (too noisy).
DEFAULT_MIN_TRADES = 20
DEFAULT_MIN_WINNERS = 10


@dataclass
class ExcursionRow:
    """
    Normalized per-trade excursion. Mirrors the fields of
    intrabar_risk.TradeExcursion that this analyzer needs, so it can accept
    either those objects or plain dicts.
    """
    realised_pnl: float
    mae: float           # worst unrealised loss, currency, <= 0
    mfe: float           # best unrealised gain, currency, >= 0
    size: float = 0.0
    symbol: str = ""

    @staticmethod
    def from_any(obj: Any) -> "ExcursionRow":
        if isinstance(obj, dict):
            g = obj.get
            return ExcursionRow(
                realised_pnl=_f(g("realised_pnl", g("realized_pnl", g("pnl", 0)))),
                mae=_f(g("mae", 0)),
                mfe=_f(g("mfe", 0)),
                size=_f(g("size", 0)),
                symbol=str(g("symbol", "")),
            )
        return ExcursionRow(
            realised_pnl=_f(getattr(obj, "realised_pnl", 0)),
            mae=_f(getattr(obj, "mae", 0)),
            mfe=_f(getattr(obj, "mfe", 0)),
            size=_f(getattr(obj, "size", 0)),
            symbol=str(getattr(obj, "symbol", "")),
        )


@dataclass
class MFEMAEReport:
    n_trades: int = 0
    n_winners: int = 0
    n_losers: int = 0
    sufficient: bool = True
    notes: List[str] = field(default_factory=list)

    # Winner MAE distribution (how far winners dipped before working out).
    winner_mae_median: Optional[float] = None
    winner_mae_p75: Optional[float] = None   # 75th percentile of |MAE|
    winner_mae_p90: Optional[float] = None
    winner_mae_max: Optional[float] = None

    # Unrealised run left on the table (MFE beyond realised gain).
    left_on_table_median: Optional[float] = None
    left_on_table_total: Optional[float] = None

    # Loser MAE vs realised loss (are stops wider than needed?).
    loser_mae_median: Optional[float] = None
    loser_realised_median: Optional[float] = None

    # Aggregate hidden adverse excursion (winners that first went underwater).
    total_hidden_adverse: float = 0.0

    def summary(self) -> str:
        L = ["MFE/MAE placement analysis",
             f"  trades={self.n_trades} winners={self.n_winners} "
             f"losers={self.n_losers}"]
        if not self.sufficient:
            L.append("  [insufficient sample -- guidance withheld]")
            for n in self.notes:
                L.append(f"  - {n}")
            return "\n".join(L)
        if self.winner_mae_p75 is not None:
            L.append(f"  winners' adverse dip: median {self.winner_mae_median:.1f}, "
                     f"p75 {self.winner_mae_p75:.1f}, p90 {self.winner_mae_p90:.1f}")
            L.append(f"    -> a stop tighter than ~{self.winner_mae_p75:.1f} "
                     f"would have cut >25% of winners (in-sample)")
        if self.left_on_table_median is not None:
            L.append(f"  run left on table (MFE beyond realised): "
                     f"median {self.left_on_table_median:.1f}, "
                     f"total {self.left_on_table_total:.1f}")
        for n in self.notes:
            L.append(f"  note: {n}")
        L.append("  (in-sample observations; validate out-of-sample before use)")
        return "\n".join(L)


class MFEMAEAnalyzer:
    """Aggregates per-trade excursions into stop/target placement statistics."""

    def __init__(self, min_trades: int = DEFAULT_MIN_TRADES,
                 min_winners: int = DEFAULT_MIN_WINNERS):
        self.min_trades = min_trades
        self.min_winners = min_winners

    def analyze(self, excursions: Sequence[Any]) -> MFEMAEReport:
        rows = [ExcursionRow.from_any(e) for e in excursions]
        report = MFEMAEReport(n_trades=len(rows))

        if len(rows) < self.min_trades:
            report.sufficient = False
            report.notes.append(
                f"only {len(rows)} trades (< {self.min_trades} required); "
                f"placement statistics would be noise")
            return report

        winners = [r for r in rows if r.realised_pnl > 0]
        losers = [r for r in rows if r.realised_pnl < 0]
        report.n_winners = len(winners)
        report.n_losers = len(losers)

        # Hidden adverse excursion (winners that first went underwater), the
        # same concept intrabar_risk exposes; summed here for a headline figure.
        report.total_hidden_adverse = sum(
            max(0.0, -r.mae - max(0.0, -r.realised_pnl)) for r in rows)

        # Winner MAE distribution: how far winners dipped (absolute currency).
        if len(winners) >= self.min_winners:
            winner_dips = sorted(abs(r.mae) for r in winners)
            report.winner_mae_median = _percentile(winner_dips, 50)
            report.winner_mae_p75 = _percentile(winner_dips, 75)
            report.winner_mae_p90 = _percentile(winner_dips, 90)
            report.winner_mae_max = winner_dips[-1]
        else:
            report.notes.append(
                f"only {len(winners)} winners (< {self.min_winners}); "
                f"winner-MAE stop guidance withheld")

        # Run left on the table: MFE beyond the realised gain, per trade.
        # For a winner realising R with peak MFE M, (M - R) is unrealised run a
        # tighter-trailing or higher target could have captured.
        left = [max(0.0, r.mfe - max(0.0, r.realised_pnl)) for r in rows]
        if left:
            report.left_on_table_median = _percentile(sorted(left), 50)
            report.left_on_table_total = sum(left)

        # Loser MAE vs realised loss.
        if losers:
            report.loser_mae_median = _percentile(
                sorted(abs(r.mae) for r in losers), 50)
            report.loser_realised_median = _percentile(
                sorted(abs(r.realised_pnl) for r in losers), 50)

        return report

    def analyze_from_trades(self, trades: Any, price_data: Any) -> MFEMAEReport:
        """
        Convenience: compute excursions via intrabar_risk, then analyze.
        Imported lazily so the pure analyzer has no hard dependency on pandas /
        intrabar_risk. Returns an insufficient report if the dependency or data
        is unavailable, rather than raising.
        """
        try:
            from intrabar_risk import trade_excursions
        except Exception as e:
            r = MFEMAEReport()
            r.sufficient = False
            r.notes.append(f"intrabar_risk unavailable: {e}")
            return r
        try:
            exc = trade_excursions(trades, price_data)
        except Exception as e:
            r = MFEMAEReport()
            r.sufficient = False
            r.notes.append(f"excursion computation failed: {e}")
            return r
        return self.analyze(exc)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile on an already-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


__all__ = ["MFEMAEAnalyzer", "MFEMAEReport", "ExcursionRow"]


if __name__ == "__main__":
    # Self-demo with synthetic excursions.
    import random
    random.seed(1)
    rows = []
    for _ in range(60):
        win = random.random() > 0.4
        if win:
            realised = random.uniform(50, 300)
            mae = -random.uniform(0, 120)      # winners dip then recover
            mfe = realised + random.uniform(0, 150)
        else:
            realised = -random.uniform(50, 200)
            mae = realised - random.uniform(0, 60)
            mfe = random.uniform(0, 80)
        rows.append({"realised_pnl": realised, "mae": mae, "mfe": mfe})
    rep = MFEMAEAnalyzer().analyze(rows)
    print(rep.summary())
