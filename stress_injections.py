# ==============================================================================
# stress_injections.py -- Monte Carlo Stress Injections
# ==============================================================================
# Hardens pass-rate estimates by stressing the daily-P&L paths BEFORE they run
# through the existing challenge simulator. The block-bootstrap MC in
# challenge_simulator / portfolio_merge preserves historical loss streaks and
# vol clustering, but the historical sample may simply not contain a bad enough
# day. These injections ask the harder question: "does the strategy still pass
# if the worst days are worse than anything in the sample, or if a shock lands?"
#
# HOW IT PLUGS IN (no change to the simulator):
#   simulate_challenge(sims, ...) already takes an (n_paths, n_days) array of
#   daily P&L. This module TRANSFORMS that array and returns a new one, so:
#
#       stressed = StressInjector(cfg).apply(sims)
#       result   = simulate_challenge(stressed, rules=rules, ...)
#
#   The injections operate on the P&L array only; they are agnostic to how the
#   bootstrap produced it and to which firm's rules will judge it.
#
# THE INJECTIONS (each independently toggleable):
#   1. Worst-day amplification -- multiply each path's single worst (most
#      negative) day by a factor (e.g. 2-3x). Models "the worst day you have
#      seen, but worse".
#   2. Shock injection -- with some probability per path, replace one random day
#      with a fixed shock loss (e.g. -X% of account). Models a gap / news event
#      not present in the sample.
#   3. Spread-doubling drag -- subtract an extra per-day cost from every day,
#      modelling spreads widening (e.g. at rollover / thin liquidity) beyond
#      what the backtest assumed. A blunt but honest proxy until a real
#      per-hour spread curve exists.
#
# DESIGN PRINCIPLE (project-wide):
#   Stressing must only ever make paths WORSE, never accidentally better. Every
#   injection is checked to be loss-directional, and apply() asserts the total
#   P&L of each path did not increase. A stress test that can flatter a strategy
#   is worse than none.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StressConfig:
    # 1. Worst-day amplification. 1.0 = off. 2.5 = worst day is 2.5x as bad.
    worst_day_factor: float = 1.0
    # How many of each path's worst days to amplify (1 = just the single worst).
    worst_day_count: int = 1

    # 2. Shock injection.
    shock_probability: float = 0.0        # per-path chance of a shock day (0=off)
    shock_loss_pct: float = 0.0           # shock day P&L = -pct * account_size
    # If a path is selected for a shock, the shock replaces its chosen day only
    # if the shock is worse than that day (never softens an already-worse day).

    # 3. Spread-doubling drag.
    extra_daily_cost: float = 0.0         # currency subtracted from every day (0=off)

    # Reproducibility.
    random_seed: int = 7


class StressInjector:
    """Transforms an (n_paths, n_days) daily-P&L array into a stressed version."""

    def __init__(self, config: Optional[StressConfig] = None,
                 account_size: float = 100_000.0):
        self.config = config or StressConfig()
        self.account_size = account_size

    def apply(self, sims: np.ndarray) -> np.ndarray:
        """
        Return a stressed copy of `sims`. The input is not modified.

        Guarantees each path's total P&L is <= its original total (stress can
        only hurt), and raises AssertionError if that invariant is ever broken.
        """
        if not isinstance(sims, np.ndarray):
            sims = np.asarray(sims, dtype=float)
        if sims.ndim != 2 or sims.size == 0:
            raise ValueError("sims must be a non-empty 2-D array of daily P&L.")

        cfg = self.config
        rng = np.random.RandomState(cfg.random_seed)
        out = sims.astype(float, copy=True)
        original_totals = out.sum(axis=1)

        if cfg.worst_day_factor != 1.0:
            out = self._amplify_worst_days(out)

        if cfg.shock_probability > 0 and cfg.shock_loss_pct > 0:
            out = self._inject_shocks(out, rng)

        if cfg.extra_daily_cost > 0:
            out = out - cfg.extra_daily_cost

        # Invariant: stress never improves a path's total P&L.
        new_totals = out.sum(axis=1)
        assert np.all(new_totals <= original_totals + 1e-6), (
            "stress injection made at least one path better -- this is a bug; "
            "stress must be loss-directional only")
        return out

    # -- Injections ------------------------------------------------------------
    def _amplify_worst_days(self, sims: np.ndarray) -> np.ndarray:
        cfg = self.config
        factor = cfg.worst_day_factor
        # Only amplify NEGATIVE days; multiplying a positive day would flatter.
        n_paths, n_days = sims.shape
        k = max(1, min(cfg.worst_day_count, n_days))
        for i in range(n_paths):
            row = sims[i]
            # Indices of the k most-negative days.
            worst_idx = np.argsort(row)[:k]
            for j in worst_idx:
                if row[j] < 0:
                    row[j] = row[j] * factor
        return sims

    def _inject_shocks(self, sims: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        cfg = self.config
        shock_value = -abs(cfg.shock_loss_pct) * self.account_size
        n_paths, n_days = sims.shape
        for i in range(n_paths):
            if rng.random_sample() < cfg.shock_probability:
                day = rng.randint(0, n_days)
                # Only replace if the shock is worse than the existing day, so
                # a shock never softens an already-catastrophic day.
                if shock_value < sims[i, day]:
                    sims[i, day] = shock_value
        return sims


def stress_and_summarize(sims: np.ndarray, config: Optional[StressConfig] = None,
                         account_size: float = 100_000.0) -> dict:
    """
    Convenience: produce a stressed array plus a small before/after summary of
    what the stress did, for logging next to a pass-rate run.
    """
    injector = StressInjector(config, account_size=account_size)
    stressed = injector.apply(sims)
    base_worst = float(sims.min())
    stressed_worst = float(stressed.min())
    return {
        "stressed": stressed,
        "n_paths": int(sims.shape[0]),
        "n_days": int(sims.shape[1]),
        "base_worst_day": base_worst,
        "stressed_worst_day": stressed_worst,
        "base_mean_total": float(sims.sum(axis=1).mean()),
        "stressed_mean_total": float(stressed.sum(axis=1).mean()),
    }


__all__ = ["StressInjector", "StressConfig", "stress_and_summarize"]


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    sims = rng.normal(50, 500, size=(1000, 20))  # 1000 paths, 20 days
    cfg = StressConfig(worst_day_factor=2.5, shock_probability=0.1,
                       shock_loss_pct=0.04, extra_daily_cost=20.0)
    summary = stress_and_summarize(sims, cfg)
    print(f"paths={summary['n_paths']} days={summary['n_days']}")
    print(f"worst day:        {summary['base_worst_day']:.0f} "
          f"-> {summary['stressed_worst_day']:.0f}")
    print(f"mean path total:  {summary['base_mean_total']:.0f} "
          f"-> {summary['stressed_mean_total']:.0f}")
