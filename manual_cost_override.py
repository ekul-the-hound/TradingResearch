# ==============================================================================
# manual_cost_override.py -- Deliberate Pessimistic Cost Override
# ==============================================================================
# Lets you set ONE deliberate, pessimistic cost assumption (spread + fee) and use
# it everywhere the cost-adjusted scorer runs -- instead of trusting the
# hardcoded defaults or trying to model every prop firm's fee schedule.
#
# THE STRATEGY (your decision):
#   Set the spread/fee slightly HIGHER than the real FTMO/broker numbers. If a
#   strategy still clears its gates under these harsher costs, it will clear the
#   real, cheaper costs on the live account. Passing a harder test guarantees
#   passing the easier one. This sidesteps the moving-target problem of modelling
#   each firm's exact fees.
#
# HOW IT FITS:
#   cost_adjusted_scoring.CostProfile already has the fields the scorer needs
#   (commission_pct, spread_pct, slippage_pct, overnight_rate, min_commission).
#   This module produces a CostProfile from your saved numbers, so nothing in the
#   scorer changes -- you just feed it this profile instead of a default.
#
# UNITS (read this once so the numbers mean what you think):
#   All rates are PERCENT OF NOTIONAL, matching CostProfile.
#     * 1 pip on EURUSD (~1.10) is about 0.009% -> ~0.01.
#     * spread_pct = 0.02  means ~2 pips of spread cost, a pessimistic majors
#       assumption (real majors are often <1 pip; 2 pips is a safe overestimate).
#   Set numbers in PIPS with pips_to_pct() if that is easier to reason about.
#
# DESIGN PRINCIPLE (project-wide):
#   The override is EXPLICIT and PERSISTED. If no override is set, this module
#   does not silently substitute a guess -- callers ask for the override and get
#   a clear "not set" answer, so cost assumptions are never invisible. When set,
#   it records exactly what you chose and when, so a backtest's cost basis is
#   always auditable.
# ==============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from discovery_config import DATA_DIR
    _DEFAULT_PATH = str(DATA_DIR / "manual_cost_override.json")
except Exception:
    _D = Path(__file__).parent / "data"
    _D.mkdir(parents=True, exist_ok=True)
    _DEFAULT_PATH = str(_D / "manual_cost_override.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# A typical majors pip as a fraction of notional, for pip<->pct conversion.
# EURUSD ~1.10: one pip (0.0001) / 1.10 ~= 0.0000909 -> 0.00909%.
PIP_PCT_MAJORS = 0.009


def pips_to_pct(pips: float, pip_pct: float = PIP_PCT_MAJORS) -> float:
    """Convert a pip count to percent-of-notional (majors default)."""
    return pips * pip_pct


def pct_to_pips(pct: float, pip_pct: float = PIP_PCT_MAJORS) -> float:
    return pct / pip_pct if pip_pct else 0.0


@dataclass
class ManualCosts:
    """
    Your deliberate pessimistic cost assumption, in percent-of-notional.

    Defaults are a conservative FTMO-forex starting point: ~2 pip spread, a small
    slippage buffer, and a tiny commission. Overnight is 0 because the strategies
    are intraday (time_stop enforces this), so swaps do not apply.
    """
    asset_class: str = "forex"
    spread_pct: float = 0.02        # ~2 pips, pessimistic for majors
    commission_pct: float = 0.002   # ~0.2 bps
    slippage_pct: float = 0.01      # ~1 pip buffer
    overnight_rate: float = 0.0     # intraday -> no swaps
    min_commission: float = 0.0
    note: str = "Pessimistic FTMO-forex override; set above real broker costs."
    updated_at: str = ""

    def total_round_trip_pct(self) -> float:
        """
        A quick headline: the cost a round-trip trade pays before financing.
        Spread is paid once crossing the book; commission + slippage apply per
        side, so this is a rough 'what each trade must overcome' figure.
        """
        return self.spread_pct + 2 * (self.commission_pct + self.slippage_pct)


class ManualCostOverride:
    """Persists a deliberate cost assumption and emits it as a CostProfile."""

    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- Set / get -------------------------------------------------------------
    def set_costs(self, costs: ManualCosts) -> ManualCosts:
        costs.updated_at = _utcnow()
        self._write(asdict(costs))
        return costs

    def set_from_pips(self, spread_pips: float, slippage_pips: float = 1.0,
                      commission_pct: float = 0.002,
                      asset_class: str = "forex") -> ManualCosts:
        """Convenience: set the override thinking in pips instead of percent."""
        costs = ManualCosts(
            asset_class=asset_class,
            spread_pct=pips_to_pct(spread_pips),
            slippage_pct=pips_to_pct(slippage_pips),
            commission_pct=commission_pct,
            overnight_rate=0.0,
            note=f"Set from pips: {spread_pips} spread, {slippage_pips} slippage "
                 f"(pessimistic FTMO-forex).",
        )
        return self.set_costs(costs)

    def is_set(self) -> bool:
        return self.path.exists()

    def get(self) -> Optional[ManualCosts]:
        data = self._read()
        if data is None:
            return None
        # Tolerate older/extra keys.
        fields = ManualCosts.__dataclass_fields__.keys()
        clean = {k: v for k, v in data.items() if k in fields}
        return ManualCosts(**clean)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    # -- Emit a CostProfile the scorer consumes --------------------------------
    def to_cost_profile(self) -> Any:
        """
        Build a cost_adjusted_scoring.CostProfile from the saved override.

        Raises if no override is set -- callers must not silently fall back to a
        default, because an invisible cost basis is exactly what this module
        exists to prevent.
        """
        costs = self.get()
        if costs is None:
            raise RuntimeError(
                "no manual cost override is set; call set_costs()/set_from_pips() "
                "first, or explicitly choose a default profile in the scorer")
        try:
            from cost_adjusted_scoring import CostProfile
        except Exception as e:
            raise RuntimeError(f"cost_adjusted_scoring unavailable: {e}") from e
        return CostProfile(
            name=f"ManualOverride({costs.asset_class})",
            commission_pct=costs.commission_pct,
            spread_pct=costs.spread_pct,
            slippage_pct=costs.slippage_pct,
            overnight_rate=costs.overnight_rate,
            min_commission=costs.min_commission,
        )

    def describe(self) -> str:
        costs = self.get()
        if costs is None:
            return "manual cost override: NOT SET"
        return (f"manual cost override [{costs.asset_class}] set {costs.updated_at}:\n"
                f"  spread     {costs.spread_pct:.4f}%  (~{pct_to_pips(costs.spread_pct):.1f} pips)\n"
                f"  slippage   {costs.slippage_pct:.4f}%  (~{pct_to_pips(costs.slippage_pct):.1f} pips)\n"
                f"  commission {costs.commission_pct:.4f}%\n"
                f"  overnight  {costs.overnight_rate:.5f}%  "
                f"({'intraday, no swaps' if costs.overnight_rate == 0 else 'holds overnight'})\n"
                f"  round-trip cost each trade must overcome: "
                f"{costs.total_round_trip_pct():.4f}%")

    # -- IO --------------------------------------------------------------------
    def _read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


__all__ = ["ManualCostOverride", "ManualCosts", "pips_to_pct", "pct_to_pips"]


if __name__ == "__main__":
    import tempfile, os
    p = tempfile.mktemp(suffix=".json")
    ov = ManualCostOverride(path=p)
    print(ov.describe())               # NOT SET
    ov.set_from_pips(spread_pips=2.0, slippage_pips=1.0)
    print()
    print(ov.describe())
    prof = ov.to_cost_profile()
    print("\nCostProfile emitted:", prof.name,
          "spread_pct=", prof.spread_pct, "overnight=", prof.overnight_rate)
    os.remove(p)
