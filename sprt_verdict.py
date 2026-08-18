# ==============================================================================
# sprt_verdict.py -- Sequential Probability Ratio Test for Demo Verdicts
# ==============================================================================
# Decides KEEP / KILL / KEEP-WATCHING for a strategy on demo (forward test) by
# comparing its unfolding results against what the backtest promised -- using
# Wald's Sequential Probability Ratio Test, which reaches a decision with the
# FEWEST trades needed to hit target error rates, rather than an arbitrary "wait
# for N trades" rule.
#
# WHAT IT ACTUALLY TESTS:
#   Two hypotheses about the strategy's true per-trade edge:
#     H0 (KEEP)  -- the edge matches the backtest expectation (p = p0).
#     H1 (KILL)  -- the edge has degraded to a "dead" level (p = p1 < p0).
#   As each demo trade lands, the log-likelihood ratio accumulates. When it
#   crosses the upper boundary we accept H1 (KILL); the lower boundary, H0
#   (KEEP); in between, KEEP-WATCHING (not enough evidence yet).
#
#   This is a CONSISTENCY test, not a quality test. A KEEP verdict means "demo
#   behaviour is consistent with the backtested edge", NOT "this strategy is
#   good". A strategy with a weak-but-real backtested edge that holds up will
#   KEEP; that is correct -- the demo did its job of catching degradation.
#
# TWO OBSERVATION MODELS:
#   * Bernoulli (win/loss)  -- classic SPRT on win rate. Feed win=True/False.
#   * Bounded returns       -- an approximate SPRT on per-trade return sign/scale
#                              via a win/loss reduction, kept explicit so its
#                              approximation is visible rather than hidden.
#
# DESIGN PRINCIPLE (project-wide):
#   The default verdict is KEEP-WATCHING -- the test refuses to declare KEEP or
#   KILL until the evidence crosses a boundary calibrated to the chosen error
#   rates. A minimum-trades floor prevents an early lucky/unlucky streak from
#   forcing a verdict before the sample deserves one. Undecided is stated
#   plainly, never rounded to a confident call.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# Verdicts.
KEEP = "KEEP"
KILL = "KILL"
WATCH = "KEEP_WATCHING"


@dataclass
class SPRTConfig:
    # H0: backtested win rate (the edge we hope holds). Must be in (0, 1).
    p0_win_rate: float = 0.55
    # H1: degraded "dead" win rate we want to catch. Must be < p0.
    p1_win_rate: float = 0.45
    # Target error rates:
    alpha: float = 0.05   # P(KILL a good strategy)  -- false kill
    beta: float = 0.05    # P(KEEP a dead strategy)  -- false keep
    # Do not emit KEEP/KILL before this many trades, regardless of boundary.
    min_trades: int = 20
    # Optional hard cap: after this many trades, force the nearer boundary's
    # verdict rather than watching forever. 0 = no cap (stay in WATCH).
    max_trades: int = 0


@dataclass
class SPRTState:
    n: int = 0
    wins: int = 0
    losses: int = 0
    log_lr: float = 0.0           # accumulated log-likelihood ratio
    verdict: str = WATCH
    reason: str = ""

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0


class SPRT:
    """
    Wald SPRT for KEEP/KILL/KEEP-WATCHING on a stream of win/loss outcomes.

    Boundaries (Wald):
        A = log((1 - beta) / alpha)     -- cross above -> accept H1 (KILL)
        B = log(beta / (1 - alpha))     -- cross below -> accept H0 (KEEP)
    Per-trade log-likelihood increment:
        win  -> log(p1 / p0)
        loss -> log((1 - p1) / (1 - p0))
    """

    def __init__(self, config: Optional[SPRTConfig] = None):
        self.config = config or SPRTConfig()
        self._validate()
        self.state = SPRTState()
        c = self.config
        self._A = math.log((1.0 - c.beta) / c.alpha)       # upper (KILL)
        self._B = math.log(c.beta / (1.0 - c.alpha))       # lower (KEEP)
        self._win_incr = math.log(c.p1_win_rate / c.p0_win_rate)
        self._loss_incr = math.log((1.0 - c.p1_win_rate) / (1.0 - c.p0_win_rate))

    def _validate(self) -> None:
        c = self.config
        if not (0.0 < c.p1_win_rate < c.p0_win_rate < 1.0):
            raise ValueError(
                "require 0 < p1_win_rate < p0_win_rate < 1 "
                f"(got p0={c.p0_win_rate}, p1={c.p1_win_rate})")
        if not (0.0 < c.alpha < 1.0 and 0.0 < c.beta < 1.0):
            raise ValueError("alpha and beta must be in (0, 1)")

    # -- Feeding outcomes ------------------------------------------------------
    def update(self, win: bool) -> SPRTState:
        """Feed one demo trade outcome. Returns the current state/verdict."""
        s = self.state
        # Once decided, stay decided (verdict is sticky).
        if s.verdict in (KEEP, KILL):
            return s

        s.n += 1
        if win:
            s.wins += 1
            s.log_lr += self._win_incr
        else:
            s.losses += 1
            s.log_lr += self._loss_incr

        self._evaluate()
        return s

    def update_return(self, trade_return: float) -> SPRTState:
        """
        Feed a per-trade return. Reduced to win/loss by sign (>0 = win). This is
        the explicit, visible approximation noted in the module docstring:
        magnitude is not used, only direction. Feed win/loss via update() if you
        want the exact Bernoulli test.
        """
        return self.update(trade_return > 0)

    def update_many(self, outcomes: List[bool]) -> SPRTState:
        for o in outcomes:
            self.update(o)
        return self.state

    # -- Decision --------------------------------------------------------------
    def _evaluate(self) -> None:
        s = self.state
        c = self.config

        # Respect the minimum-trades floor for KEEP/KILL.
        if s.n < c.min_trades:
            s.verdict = WATCH
            s.reason = (f"only {s.n} trades (< {c.min_trades} floor); "
                        f"watching")
            return

        if s.log_lr >= self._A:
            s.verdict = KILL
            s.reason = (f"log-LR {s.log_lr:.2f} crossed KILL boundary "
                        f"{self._A:.2f}: demo win rate {s.win_rate:.2%} is "
                        f"consistent with the degraded hypothesis")
            return
        if s.log_lr <= self._B:
            s.verdict = KEEP
            s.reason = (f"log-LR {s.log_lr:.2f} crossed KEEP boundary "
                        f"{self._B:.2f}: demo win rate {s.win_rate:.2%} is "
                        f"consistent with the backtested edge")
            return

        # Undecided. Honour a hard cap if configured.
        if c.max_trades and s.n >= c.max_trades:
            # Force the nearer boundary's verdict.
            dist_kill = abs(self._A - s.log_lr)
            dist_keep = abs(s.log_lr - self._B)
            if dist_kill <= dist_keep:
                s.verdict = KILL
                s.reason = (f"max_trades {c.max_trades} reached undecided; "
                            f"forcing nearer boundary -> KILL")
            else:
                s.verdict = KEEP
                s.reason = (f"max_trades {c.max_trades} reached undecided; "
                            f"forcing nearer boundary -> KEEP")
            return

        s.verdict = WATCH
        s.reason = (f"{s.n} trades, log-LR {s.log_lr:.2f} between boundaries "
                    f"[{self._B:.2f}, {self._A:.2f}]; watching")

    # -- Introspection ---------------------------------------------------------
    def boundaries(self) -> dict:
        return {"keep_boundary": self._B, "kill_boundary": self._A,
                "win_increment": self._win_incr,
                "loss_increment": self._loss_incr}

    def reset(self) -> None:
        self.state = SPRTState()


def verdict_from_outcomes(outcomes: List[bool],
                          config: Optional[SPRTConfig] = None) -> SPRTState:
    """Convenience: run a full sequence and return the final state."""
    sprt = SPRT(config)
    sprt.update_many(outcomes)
    return sprt.state


__all__ = ["SPRT", "SPRTConfig", "SPRTState", "verdict_from_outcomes",
           "KEEP", "KILL", "WATCH"]


if __name__ == "__main__":
    # Deterministic streams so the demo is reproducible, not RNG-dependent.
    good = ([True, True, True, False, False] * 40)  # 60% wins -> KEEP
    s = verdict_from_outcomes(good, SPRTConfig(min_trades=20))
    print(f"held edge (60%): {s.verdict} after {s.n} trades "
          f"(win rate {s.win_rate:.2%})")

    bad = ([True, True, False, False, False] * 40)  # 40% wins -> KILL
    s = verdict_from_outcomes(bad, SPRTConfig(min_trades=20))
    print(f"degraded (40%):  {s.verdict} after {s.n} trades "
          f"(win rate {s.win_rate:.2%})")
