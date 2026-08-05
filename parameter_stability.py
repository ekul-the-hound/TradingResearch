# ==============================================================================
# parameter_stability.py
# ==============================================================================
# Phase 2, Item 16 -- parameter-stability plateau as a hard promotion gate.
#
# WHY THIS IS A GATE AND NOT A METRIC
# -----------------------------------
# In a search-heavy pipeline, the best-scoring parameter set is usually the
# luckiest one, not the best one. Out of a few hundred combinations, some will
# land on a noise spike -- a narrow peak that exists because that specific
# combination happened to catch a handful of favourable trades. It looks
# excellent and reproduces perfectly on the same data, then evaporates live,
# because live prices are not the sample the spike was fitted to.
#
# The defence is not a better optimiser. It is refusing to promote a point that
# does not sit on a plateau. A parameter set whose neighbours perform almost as
# well is describing something the market actually does; a set whose neighbours
# collapse is describing this dataset.
#
# parameter_sensitivity.py already computes stability_score, plateau_score and
# cliff_score. Nothing gates on them -- the same orphaned-capability pattern the
# lookahead detector had before it was wired in.
#
# WHY NOT JUST THRESHOLD THE EXISTING SCORE
# -----------------------------------------
# Because it cannot be thresholded. stability_score is:
#
#     np.mean(np.abs(np.diff(returns)))
#
# which is unnormalised, so it scales with the magnitude of the returns:
#
#     low-return, JAGGED   [1.0, 0.2, 1.1, 0.1, 1.2]  -> 0.95
#     high-return, SMOOTH  [40, 42, 44, 46, 48]       -> 2.00
#
# Lower is supposed to mean more stable. A fixed threshold ranks these
# backwards, and would systematically pass low-return strategies while failing
# high-return ones regardless of actual stability. Every metric here is
# scale-free by construction, and there is a test that pins that property.
#
# WHAT IS MEASURED (all dimensionless)
# ------------------------------------
#   plateau_ratio     share of the neighbourhood retaining >= `retain` of the
#                     chosen point's score
#   spike_index       how far the chosen point stands above its neighbours,
#                     relative to the local spread. 0 = flat, 1 = isolated peak
#   sign_consistency  share of the neighbourhood still profitable. Blunt, and
#                     the hardest to fake
#   cliff_distance    steps from the chosen point before the score falls below
#                     `floor`, in units of parameter steps
#   roughness         mean |diff| divided by mean |level| -- the scale-free
#                     replacement for stability_score
#
# EVALUATE THE POINT YOU WILL TRADE
# ---------------------------------
# The gate measures the CHOSEN point, not the best one. Optimising to the peak
# and then measuring stability at the peak answers the wrong question. And
# because the peak is usually a spike, recommend_robust_point() returns the
# plateau centre instead -- typically a slightly lower score that survives.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from numpy.typing import ArrayLike

import numpy as np

# Defaults. Deliberately demanding: a gate that passes almost everything is
# decoration, and the cost of a false negative here is one discarded strategy
# while the cost of a false positive is a failed challenge.
DEFAULTS = {
    'neighbourhood': 2,        # steps either side of the chosen point
    'retain': 0.60,            # neighbour must keep 60% of the chosen score
    'min_plateau_ratio': 0.60,
    'max_spike_index': 0.35,   # own-magnitude scale: plateaus ~0.1, spikes ~1.0
    'min_sign_consistency': 0.80,
    'min_cliff_distance': 2,
    'floor': 0.0,              # score below this counts as falling off a cliff
}


@dataclass
class StabilityReport:
    chosen_index: Any
    chosen_score: float
    plateau_ratio: float = 0.0
    spike_index: float = 1.0
    sign_consistency: float = 0.0
    cliff_distance: float = 0.0
    roughness: float = 0.0
    neighbourhood_size: int = 0
    failures: List[str] = field(default_factory=list)
    recommended_index: Any = None
    recommended_score: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return not self.failures and self.error is None

    def summary(self) -> str:
        L = [f"\n{'=' * 66}", "  PARAMETER STABILITY", '=' * 66]
        if self.error:
            L += [f"  [ERROR] {self.error}", '=' * 66]
            return '\n'.join(L)
        L.append(f"  Chosen point:      {self.chosen_index}  "
                 f"score {self.chosen_score:.4f}")
        L.append(f"  Neighbourhood:     {self.neighbourhood_size} points")
        L.append("")
        L.append(f"  plateau_ratio      {self.plateau_ratio:6.2f}   "
                 f"(share of neighbours holding up)")
        L.append(f"  spike_index        {self.spike_index:6.2f}   "
                 f"(0 = flat, 1 = isolated peak)")
        L.append(f"  sign_consistency   {self.sign_consistency:6.2f}   "
                 f"(share still profitable)")
        L.append(f"  cliff_distance     {self.cliff_distance:6.2f}   "
                 f"(steps before falling through the floor)")
        L.append(f"  roughness          {self.roughness:6.2f}   "
                 f"(scale-free; lower is smoother)")
        if self.recommended_index is not None and self.recommended_index != self.chosen_index:
            L.append("")
            L.append(f"  Plateau centre is {self.recommended_index} "
                     f"(score {self.recommended_score:.4f}).")
            L.append("  A slightly lower score that survives beats a peak that does not.")
        L.append("")
        if self.passed:
            L.append("  VERDICT: PASS - the chosen point sits on a plateau")
        else:
            L.append(f"  VERDICT: FAIL - {len(self.failures)} criterion/criteria unmet:")
            for f in self.failures:
                L.append(f"    - {f}")
        L.append('=' * 66)
        return '\n'.join(L)


# ==============================================================================
# METRICS
# ==============================================================================

def _neighbour_slice(scores: np.ndarray, idx: int, k: int) -> np.ndarray:
    lo, hi = max(0, idx - k), min(len(scores), idx + k + 1)
    return scores[lo:hi]


def roughness(scores: ArrayLike) -> float:
    """
    Mean absolute step divided by mean absolute level.

    The scale-free replacement for stability_score. Dividing by the level is
    the whole point: without it, a smooth curve at 40% return scores worse than
    a jagged one at 1%.
    """
    a = np.asarray(scores, dtype=float)
    if a.size < 2:
        return 0.0
    level = float(np.mean(np.abs(a)))
    if level < 1e-12:
        return 0.0
    return float(np.mean(np.abs(np.diff(a))) / level)


def spike_index(scores: ArrayLike, idx: int, k: int = 2) -> float:
    """
    How far the chosen point stands above its neighbours, as a fraction of its
    own magnitude.

    0.0  neighbours match it -- a plateau
    1.0  neighbours are at zero -- an isolated spike

    NORMALISED BY THE POINT'S OWN MAGNITUDE, NOT THE LOCAL SPREAD. An earlier
    version divided by (max - min) of the neighbourhood, which turned out to be
    near-constant for any unimodal shape -- the peak is always at the top of
    its own local range, so a broad plateau scored 0.67 and a gentle hill 0.75,
    against a sharp spike at 1.00. It barely separated the cases it existed to
    separate, and a gentle hill sat on the threshold by construction.

    Dividing by |chosen| gives 0.11 / 0.04 / 0.96 for the same three shapes,
    and stays scale-free because numerator and denominator scale together.
    """
    a = np.asarray(scores, dtype=float)
    nb = _neighbour_slice(a, idx, k)
    if nb.size < 2:
        return 1.0
    chosen = float(a[idx])
    hit = np.where(np.isclose(nb, chosen))[0]
    others = np.delete(nb, hit[:1]) if hit.size else nb
    if others.size == 0:
        return 1.0
    scale = abs(chosen)
    if scale < 1e-12:
        # A chosen score of ~0 makes a relative measure meaningless. Treat any
        # separation from the neighbours as total rather than dividing by zero.
        return 0.0 if np.allclose(others, chosen) else 1.0
    return float(np.clip((chosen - np.median(others)) / scale, 0.0, 1.0))


def plateau_ratio(scores: ArrayLike, idx: int, k: int = 2,
                  retain: float = 0.60) -> float:
    """Share of the neighbourhood retaining `retain` of the chosen score."""
    a = np.asarray(scores, dtype=float)
    nb = _neighbour_slice(a, idx, k)
    chosen = a[idx]
    if nb.size == 0:
        return 0.0
    if chosen <= 0:
        # A non-positive chosen score makes "retain 60% of it" meaningless --
        # 60% of a loss is a smaller loss. Fall back to sign agreement.
        return float(np.mean(nb <= 0))
    return float(np.mean(nb >= chosen * retain))


def sign_consistency(scores: ArrayLike, idx: int, k: int = 2,
                     floor: float = 0.0) -> float:
    """Share of the neighbourhood still above the floor. Blunt and hard to fake."""
    a = np.asarray(scores, dtype=float)
    nb = _neighbour_slice(a, idx, k)
    return float(np.mean(nb > floor)) if nb.size else 0.0


def cliff_distance(scores: ArrayLike, idx: int, floor: float = 0.0) -> float:
    """
    Steps from the chosen point in the worse direction before the score drops
    through the floor. Reported as the smaller of the two directions, because a
    plateau with a cliff on one side is still a cliff.
    """
    a = np.asarray(scores, dtype=float)
    n = len(a)

    def walk(step):
        d, i = 0, idx
        while 0 <= i + step < n:
            i += step
            d += 1
            if a[i] <= floor:
                return d - 1
        return d

    return float(min(walk(1), walk(-1)))


def recommend_robust_point(scores: ArrayLike, k: int = 2,
                           retain: float = 0.60) -> Tuple[int, float]:
    """
    The plateau centre: the index whose neighbourhood minimum is highest.

    Maximising the worst neighbour rather than the point itself is what makes
    this robust -- it picks the middle of a broad hill over the tip of a narrow
    one, which is exactly the trade a search will not make on its own.
    """
    a = np.asarray(scores, dtype=float)
    if a.size == 0:
        return 0, 0.0
    best_i, best_worst = 0, -np.inf
    for i in range(a.size):
        worst = float(np.min(_neighbour_slice(a, i, k)))
        if worst > best_worst:
            best_i, best_worst = i, worst
    return best_i, float(a[best_i])


# ==============================================================================
# GATE
# ==============================================================================

def analyze_1d(scores: ArrayLike, chosen_index: Optional[int] = None,
               thresholds: Optional[Dict[str, Any]] = None) -> StabilityReport:
    """
    Evaluate one parameter sweep.

    scores: performance at each parameter value, in parameter order.
    chosen_index: the point you intend to trade. Defaults to the best-scoring
                  one -- which is usually the spike, and is exactly why the
                  recommendation is reported alongside.
    """
    th = dict(DEFAULTS)
    th.update(thresholds or {})

    a = np.asarray(scores, dtype=float)
    if a.size == 0:
        return StabilityReport(chosen_index=None, chosen_score=0.0,
                               error="No scores supplied")
    if a.size < 3:
        return StabilityReport(
            chosen_index=chosen_index, chosen_score=float(a[0]),
            error=f"Need at least 3 sweep points to judge a plateau, got {a.size}")
    if not np.all(np.isfinite(a)):
        return StabilityReport(
            chosen_index=chosen_index, chosen_score=0.0,
            error="Scores contain NaN or inf -- a failed backtest must not be "
                  "read as a flat neighbourhood")

    idx = int(np.argmax(a)) if chosen_index is None else int(chosen_index)
    if not 0 <= idx < a.size:
        return StabilityReport(chosen_index=chosen_index, chosen_score=0.0,
                               error=f"chosen_index {chosen_index} out of range")

    k = int(th['neighbourhood'])
    rep = StabilityReport(chosen_index=idx, chosen_score=float(a[idx]))
    rep.neighbourhood_size = int(_neighbour_slice(a, idx, k).size)
    rep.plateau_ratio = plateau_ratio(a, idx, k, th['retain'])
    rep.spike_index = spike_index(a, idx, k)
    rep.sign_consistency = sign_consistency(a, idx, k, th['floor'])
    rep.cliff_distance = cliff_distance(a, idx, th['floor'])
    rep.roughness = roughness(a)
    rep.recommended_index, rep.recommended_score = recommend_robust_point(
        a, k, th['retain'])

    if rep.plateau_ratio < th['min_plateau_ratio']:
        rep.failures.append(
            f"plateau_ratio {rep.plateau_ratio:.2f} < {th['min_plateau_ratio']:.2f} "
            f"-- too few neighbours hold up")
    if rep.spike_index > th['max_spike_index']:
        rep.failures.append(
            f"spike_index {rep.spike_index:.2f} > {th['max_spike_index']:.2f} "
            f"-- the point stands isolated above its neighbours")
    if rep.sign_consistency < th['min_sign_consistency']:
        rep.failures.append(
            f"sign_consistency {rep.sign_consistency:.2f} < "
            f"{th['min_sign_consistency']:.2f} -- neighbours are unprofitable")
    if rep.cliff_distance < th['min_cliff_distance']:
        rep.failures.append(
            f"cliff_distance {rep.cliff_distance:.0f} < {th['min_cliff_distance']} "
            f"-- a small parameter change falls off a cliff")
    return rep


def analyze_2d(matrix, chosen: Optional[Tuple[int, int]] = None,
               thresholds: Optional[Dict[str, Any]] = None) -> StabilityReport:
    """
    Evaluate a two-parameter surface.

    The neighbourhood is the surrounding block rather than a line, so a point
    that is stable along one axis and a cliff along the other is correctly
    rejected -- which a pair of independent 1D sweeps would miss.
    """
    th = dict(DEFAULTS)
    th.update(thresholds or {})

    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.size == 0:
        return StabilityReport(chosen_index=None, chosen_score=0.0,
                               error="Expected a non-empty 2D score matrix")
    if not np.all(np.isfinite(m)):
        return StabilityReport(chosen_index=None, chosen_score=0.0,
                               error="Matrix contains NaN or inf")

    if chosen is None:
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    else:
        i, j = chosen
    i, j = int(i), int(j)
    if not (0 <= i < m.shape[0] and 0 <= j < m.shape[1]):
        return StabilityReport(chosen_index=chosen, chosen_score=0.0,
                               error=f"chosen {chosen} out of range")

    k = int(th['neighbourhood'])
    block = m[max(0, i - k):i + k + 1, max(0, j - k):j + k + 1].ravel()
    chosen_score = float(m[i, j])

    rep = StabilityReport(chosen_index=(i, j), chosen_score=chosen_score)
    rep.neighbourhood_size = int(block.size)
    rep.plateau_ratio = (float(np.mean(block >= chosen_score * th['retain']))
                         if chosen_score > 0 else float(np.mean(block <= 0)))
    # Same own-magnitude normalisation as the 1D case; see spike_index().
    others = block[~np.isclose(block, chosen_score)]
    scale = abs(chosen_score)
    if others.size == 0:
        rep.spike_index = 1.0
    elif scale < 1e-12:
        rep.spike_index = 0.0 if np.allclose(others, chosen_score) else 1.0
    else:
        rep.spike_index = float(np.clip(
            (chosen_score - float(np.median(others))) / scale, 0.0, 1.0))
    rep.sign_consistency = float(np.mean(block > th['floor']))
    rep.roughness = roughness(m.ravel())
    rep.cliff_distance = float(min(
        cliff_distance(m[i, :], j, th['floor']),
        cliff_distance(m[:, j], i, th['floor'])))

    bi, bj, best_worst = i, j, -np.inf
    for a_ in range(m.shape[0]):
        for b_ in range(m.shape[1]):
            nb = m[max(0, a_ - k):a_ + k + 1, max(0, b_ - k):b_ + k + 1]
            w = float(np.min(nb))
            if w > best_worst:
                bi, bj, best_worst = a_, b_, w
    rep.recommended_index = (bi, bj)
    rep.recommended_score = float(m[bi, bj])

    if rep.plateau_ratio < th['min_plateau_ratio']:
        rep.failures.append(f"plateau_ratio {rep.plateau_ratio:.2f} < "
                            f"{th['min_plateau_ratio']:.2f}")
    if rep.spike_index > th['max_spike_index']:
        rep.failures.append(f"spike_index {rep.spike_index:.2f} > "
                            f"{th['max_spike_index']:.2f}")
    if rep.sign_consistency < th['min_sign_consistency']:
        rep.failures.append(f"sign_consistency {rep.sign_consistency:.2f} < "
                            f"{th['min_sign_consistency']:.2f}")
    if rep.cliff_distance < th['min_cliff_distance']:
        rep.failures.append(f"cliff_distance {rep.cliff_distance:.0f} < "
                            f"{th['min_cliff_distance']}")
    return rep


# ==============================================================================
# IRREGULAR SCATTER
# ==============================================================================
# analyze_1d and analyze_2d assume a grid: parameters swept at even steps. Real
# evidence in this pipeline is not on a grid. The mutation loop produces
# variants at scattered parameter values, and multi_objective_optimizer works
# over a pool of already-backtested strategies described by fingerprints -- it
# never varies a parameter itself, so there is no sweep to read.
#
# analyze_scatter answers the same question on irregularly sampled points: does
# the chosen point sit among neighbours that also work, or is it alone?
#
# WHAT IT MEASURES DEPENDS ON WHAT YOU FEED IT
#   parameter vectors  -> genuine parameter stability
#   fingerprints       -> whether a strategy is an isolated outlier in
#                         behaviour space. Related and useful, but NOT the same
#                         claim. Two strategies with near-identical behaviour
#                         can have unrelated parameters, so a pass here does
#                         not establish that the parameters are robust.
#
# Worth keeping straight, because reporting the second as though it were the
# first is precisely the kind of overclaim this codebase keeps producing.


def _normalise_points(points: np.ndarray) -> np.ndarray:
    """
    Scale each dimension to unit range before measuring distance.

    Without this a parameter measured in bars (10-200) swamps one measured as a
    fraction (0.01-0.05), and "nearest neighbour" silently means "nearest in
    the widest parameter". Constant dimensions are left at zero rather than
    dividing by their zero range.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = hi - lo
    out = np.zeros_like(p)
    live = span > 1e-12
    out[:, live] = (p[:, live] - lo[live]) / span[live]
    return out


def analyze_scatter(points, scores, chosen_index: Optional[int] = None,
                    k_neighbours: int = 5,
                    thresholds: Optional[Dict[str, Any]] = None) -> StabilityReport:
    """
    Plateau analysis on irregularly sampled points.

    Args:
        points: (N, D) parameter vectors or fingerprints.
        scores: (N,) performance for each point.
        chosen_index: the point you intend to trade. Defaults to the best.
        k_neighbours: how many nearest points form the neighbourhood.

    cliff_distance is not defined without an ordered axis, so it is reported as
    the count of neighbours above the floor rather than a step count, and the
    corresponding threshold is not applied. Reporting a step count here would
    be inventing a number the geometry does not contain.
    """
    th = dict(DEFAULTS)
    th.update(thresholds or {})

    a = np.asarray(scores, dtype=float)
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    if a.size == 0 or p.size == 0:
        return StabilityReport(chosen_index=None, chosen_score=0.0,
                               error="No points supplied")
    if p.shape[0] != a.size:
        return StabilityReport(chosen_index=None, chosen_score=0.0,
                               error=f"{p.shape[0]} points but {a.size} scores")
    if a.size < 4:
        return StabilityReport(
            chosen_index=chosen_index, chosen_score=float(a[0]),
            error=f"Need at least 4 points to judge a neighbourhood, got {a.size}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(p)):
        return StabilityReport(
            chosen_index=chosen_index, chosen_score=0.0,
            error="Points or scores contain NaN/inf -- a failed backtest must "
                  "not be read as a well-behaved neighbourhood")

    idx = int(np.argmax(a)) if chosen_index is None else int(chosen_index)
    if not 0 <= idx < a.size:
        return StabilityReport(chosen_index=chosen_index, chosen_score=0.0,
                               error=f"chosen_index {chosen_index} out of range")

    norm = _normalise_points(p)
    dists = np.linalg.norm(norm - norm[idx], axis=1)
    order = np.argsort(dists)
    k = min(int(k_neighbours) + 1, a.size)
    nb_idx = order[:k]
    nb = a[nb_idx]

    chosen = float(a[idx])
    rep = StabilityReport(chosen_index=idx, chosen_score=chosen)
    rep.neighbourhood_size = int(nb.size)
    rep.plateau_ratio = (float(np.mean(nb >= chosen * th['retain']))
                         if chosen > 0 else float(np.mean(nb <= 0)))
    others = nb[nb_idx != idx] if nb.size > 1 else nb
    scale = abs(chosen)
    if others.size == 0:
        rep.spike_index = 1.0
    elif scale < 1e-12:
        rep.spike_index = 0.0 if np.allclose(others, chosen) else 1.0
    else:
        rep.spike_index = float(np.clip(
            (chosen - float(np.median(others))) / scale, 0.0, 1.0))
    rep.sign_consistency = float(np.mean(nb > th['floor']))
    rep.roughness = roughness(a[order])
    rep.cliff_distance = float(np.sum(nb > th['floor']))

    # The plateau centre: the point whose neighbourhood holds up worst-case.
    best_i, best_worst = idx, -np.inf
    for i in range(a.size):
        d = np.linalg.norm(norm - norm[i], axis=1)
        w = float(np.min(a[np.argsort(d)[:k]]))
        if w > best_worst:
            best_i, best_worst = i, w
    rep.recommended_index, rep.recommended_score = best_i, float(a[best_i])

    if rep.plateau_ratio < th['min_plateau_ratio']:
        rep.failures.append(
            f"plateau_ratio {rep.plateau_ratio:.2f} < {th['min_plateau_ratio']:.2f} "
            f"-- too few neighbours hold up")
    if rep.spike_index > th['max_spike_index']:
        rep.failures.append(
            f"spike_index {rep.spike_index:.2f} > {th['max_spike_index']:.2f} "
            f"-- isolated among its neighbours")
    if rep.sign_consistency < th['min_sign_consistency']:
        rep.failures.append(
            f"sign_consistency {rep.sign_consistency:.2f} < "
            f"{th['min_sign_consistency']:.2f} -- neighbours are unprofitable")
    return rep


def gate(scores, chosen_index=None, thresholds=None, verbose: bool = False) -> bool:
    """
    True if the chosen parameter point may be promoted.

    A sweep too short to judge returns False, not True. "Could not measure" is
    not "measured and fine" -- promoting on an unmeasured plateau is the thing
    this exists to stop.
    """
    a = np.asarray(scores, dtype=float)
    rep = (analyze_2d(a, chosen_index, thresholds) if a.ndim == 2
           else analyze_1d(a, chosen_index, thresholds))
    if verbose:
        print(rep.summary())
    return rep.passed