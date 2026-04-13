# ==============================================================================
# genetic_operators.py
# ==============================================================================
# Phase 2, Module 5 (Week 9): Genetic Algorithm Operators
#
# Building blocks for evolutionary strategy optimization:
#   1. Selection: Tournament, roulette wheel, rank-based
#   2. Crossover: Parameter blend, entry/exit swap, indicator mix
#   3. Mutation: Parameter perturbation, indicator add/remove, type shift
#   4. Population management: elitism, diversity enforcement, archive
#
# These operators work on strategy PARAMETER VECTORS (fingerprints),
# not on source code directly. Source code generation is handled by
# mutate_strategy.py using Claude API -- these operators manipulate
# the numeric parameter space that gets translated into mutation specs.
#
# GitHub repos:
#   - pymoo (operators infrastructure)
#   - scikit-learn (distance metrics)
#
# Consumed by:
#   - multi_objective_optimizer.py (NSGA-II uses these operators)
#   - optimization_pipeline.py (generation loop)
#
# Usage:
#     from genetic_operators import GeneticEngine
#
#     engine = GeneticEngine(pop_size=100)
#     parents = engine.select(population, fitness, n=50)
#     children = engine.crossover(parents)
#     mutated = engine.mutate(children, generation=5)
#     new_pop = engine.survive(population + mutated, fitness_all)
#
# ==============================================================================

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class GeneticConfig:
    """GA hyperparameters."""
    pop_size: int = 100
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    mutation_scale: float = 0.1         # Std dev of Gaussian perturbation
    elite_frac: float = 0.1            # Top 10% survive unconditionally
    diversity_threshold: float = 0.05   # Min distance between pop members
    archive_size: int = 200             # Max Pareto archive size
    adaptive_mutation: bool = True      # Scale mutation by generation
    random_seed: int = 42


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class GenerationResult:
    """Stats from one generation of evolution."""
    generation: int
    pop_size: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float             # Mean pairwise distance
    n_unique: int
    pareto_size: int
    mutation_rate: float
    crossover_rate: float


# ==============================================================================
# GENETIC ENGINE
# ==============================================================================

class GeneticEngine:
    """
    Evolutionary operators for strategy optimization.

    Operates on population of strategy parameter vectors (np.ndarray).
    """

    def __init__(self, config: Optional[GeneticConfig] = None):
        self.config = config or GeneticConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        self.archive: List[np.ndarray] = []
        self.archive_fitness: List[float] = []
        self.generation = 0

    # ------------------------------------------------------------------
    # SELECTION
    # ------------------------------------------------------------------
    def select_tournament(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """Tournament selection: pick best from random subset."""
        selected_idx = []
        n = len(population)
        for _ in range(n_select):
            candidates = self.rng.choice(n, self.config.tournament_size, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            selected_idx.append(winner)
        return population[selected_idx]

    def select_roulette(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """Roulette wheel: probability proportional to fitness."""
        shifted = fitness - fitness.min() + 1e-6
        probs = shifted / shifted.sum()
        idx = self.rng.choice(len(population), n_select, p=probs, replace=True)
        return population[idx]

    def select_rank(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """Rank-based selection: selection pressure based on rank, not value."""
        ranks = np.argsort(np.argsort(fitness)) + 1  # 1-indexed
        probs = ranks / ranks.sum()
        idx = self.rng.choice(len(population), n_select, p=probs, replace=True)
        return population[idx]

    # ------------------------------------------------------------------
    # CROSSOVER
    # ------------------------------------------------------------------
    def crossover_blend(
        self,
        parents: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """BLX-α crossover: blend parent parameters."""
        n = len(parents)
        children = []
        indices = self.rng.permutation(n)
        for i in range(0, n - 1, 2):
            if self.rng.random() < self.config.crossover_prob:
                p1, p2 = parents[indices[i]], parents[indices[i + 1]]
                d = np.abs(p1 - p2)
                lo = np.minimum(p1, p2) - alpha * d
                hi = np.maximum(p1, p2) + alpha * d
                c1 = self.rng.uniform(lo, hi)
                c2 = self.rng.uniform(lo, hi)
                children.extend([c1, c2])
            else:
                children.extend([parents[indices[i]].copy(),
                                 parents[indices[i + 1]].copy()])
        if n % 2 == 1:
            children.append(parents[indices[-1]].copy())
        return np.array(children)

    def crossover_uniform(
        self,
        parents: np.ndarray,
    ) -> np.ndarray:
        """Uniform crossover: randomly pick each gene from one parent."""
        n = len(parents)
        children = []
        indices = self.rng.permutation(n)
        for i in range(0, n - 1, 2):
            if self.rng.random() < self.config.crossover_prob:
                p1, p2 = parents[indices[i]], parents[indices[i + 1]]
                mask = self.rng.random(len(p1)) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
                children.extend([c1, c2])
            else:
                children.extend([parents[indices[i]].copy(),
                                 parents[indices[i + 1]].copy()])
        if n % 2 == 1:
            children.append(parents[indices[-1]].copy())
        return np.array(children)

    def crossover_sbx(
        self,
        parents: np.ndarray,
        eta: float = 15.0,
    ) -> np.ndarray:
        """Simulated Binary Crossover (SBX) -- standard in NSGA-II."""
        n = len(parents)
        children = []
        indices = self.rng.permutation(n)
        for i in range(0, n - 1, 2):
            p1, p2 = parents[indices[i]], parents[indices[i + 1]]
            if self.rng.random() < self.config.crossover_prob:
                c1, c2 = np.copy(p1), np.copy(p2)
                for j in range(len(p1)):
                    if self.rng.random() < 0.5:
                        if abs(p1[j] - p2[j]) > 1e-14:
                            if p1[j] < p2[j]:
                                y1, y2 = p1[j], p2[j]
                            else:
                                y1, y2 = p2[j], p1[j]
                            u = self.rng.random()
                            beta = (2.0 * u) ** (1.0 / (eta + 1)) if u <= 0.5 \
                                else (1.0 / (2.0 * (1 - u))) ** (1.0 / (eta + 1))
                            c1[j] = 0.5 * ((1 + beta) * y1 + (1 - beta) * y2)
                            c2[j] = 0.5 * ((1 - beta) * y1 + (1 + beta) * y2)
                children.extend([c1, c2])
            else:
                children.extend([p1.copy(), p2.copy()])
        if n % 2 == 1:
            children.append(parents[indices[-1]].copy())
        return np.array(children)

    # ------------------------------------------------------------------
    # MUTATION
    # ------------------------------------------------------------------
    def mutate_gaussian(
        self,
        population: np.ndarray,
        generation: Optional[int] = None,
    ) -> np.ndarray:
        """Gaussian mutation: perturb each gene with probability p."""
        result = population.copy()
        scale = self.config.mutation_scale

        # Adaptive: reduce mutation over generations
        if self.config.adaptive_mutation and generation is not None:
            scale *= max(0.1, 1.0 - generation / 200.0)

        for i in range(len(result)):
            mask = self.rng.random(len(result[i])) < self.config.mutation_prob
            noise = self.rng.normal(0, scale, len(result[i]))
            result[i] += mask * noise
        return result

    def mutate_polynomial(
        self,
        population: np.ndarray,
        eta: float = 20.0,
    ) -> np.ndarray:
        """Polynomial mutation (standard in NSGA-II)."""
        result = population.copy()
        for i in range(len(result)):
            for j in range(len(result[i])):
                if self.rng.random() < self.config.mutation_prob:
                    u = self.rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    result[i][j] += delta * 0.1
        return result

    # ------------------------------------------------------------------
    # SURVIVAL / ELITISM
    # ------------------------------------------------------------------
    def survive(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        target_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select survivors for next generation.
        Preserves elite, fills rest by fitness.
        """
        if target_size is None:
            target_size = self.config.pop_size

        n = len(population)
        if n <= target_size:
            return population, fitness

        # Elitism: top fraction survives unconditionally
        n_elite = max(1, int(target_size * self.config.elite_frac))
        elite_idx = np.argsort(-fitness)[:n_elite]

        # Fill rest with tournament from remaining
        remaining_idx = np.setdiff1d(np.arange(n), elite_idx)
        n_fill = target_size - n_elite
        if n_fill > 0 and len(remaining_idx) > 0:
            fill = []
            for _ in range(n_fill):
                cands = self.rng.choice(remaining_idx,
                                        min(self.config.tournament_size, len(remaining_idx)),
                                        replace=False)
                winner = cands[np.argmax(fitness[cands])]
                fill.append(winner)
            survivor_idx = np.concatenate([elite_idx, fill])
        else:
            survivor_idx = elite_idx[:target_size]

        return population[survivor_idx], fitness[survivor_idx]

    # ------------------------------------------------------------------
    # DIVERSITY
    # ------------------------------------------------------------------
    def compute_diversity(self, population: np.ndarray) -> float:
        """Mean pairwise Euclidean distance."""
        n = len(population)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(min(n, 100)):
            for j in range(i + 1, min(n, 100)):
                total += np.linalg.norm(population[i] - population[j])
                count += 1
        return total / max(count, 1)

    def enforce_diversity(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove near-duplicates, keeping the fitter one."""
        keep = [0]
        for i in range(1, len(population)):
            is_unique = True
            for j in keep:
                if np.linalg.norm(population[i] - population[j]) < self.config.diversity_threshold:
                    if fitness[i] > fitness[j]:
                        keep.remove(j)
                        keep.append(i)
                    is_unique = False
                    break
            if is_unique:
                keep.append(i)
        return population[keep], fitness[keep]

    # ------------------------------------------------------------------
    # ARCHIVE (Pareto)
    # ------------------------------------------------------------------
    def update_archive(
        self,
        new_individuals: np.ndarray,
        new_fitness: np.ndarray,
    ):
        """Add non-dominated solutions to archive."""
        for ind, fit in zip(new_individuals, new_fitness):
            dominated = False
            to_remove = []
            for k, (a, af) in enumerate(zip(self.archive, self.archive_fitness)):
                if fit <= af:
                    dominated = True
                    break
                if af <= fit:
                    to_remove.append(k)
            if not dominated:
                for k in sorted(to_remove, reverse=True):
                    self.archive.pop(k)
                    self.archive_fitness.pop(k)
                self.archive.append(ind.copy())
                self.archive_fitness.append(fit)

        # Trim archive to max size
        if len(self.archive) > self.config.archive_size:
            idx = np.argsort(self.archive_fitness)[-self.config.archive_size:]
            self.archive = [self.archive[i] for i in idx]
            self.archive_fitness = [self.archive_fitness[i] for i in idx]

    # ------------------------------------------------------------------
    # FULL GENERATION STEP
    # ------------------------------------------------------------------
    def evolve_generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        generation: int,
    ) -> Tuple[np.ndarray, GenerationResult]:
        """
        Run one full generation: select -> crossover -> mutate -> survive.
        Returns new population (unevaluated) and generation stats.
        """
        self.generation = generation
        n = len(population)

        # Select parents
        n_parents = max(4, n)
        parents = self.select_tournament(population, fitness, n_parents)

        # Crossover
        children = self.crossover_sbx(parents)

        # Mutate
        children = self.mutate_gaussian(children, generation)

        # Stats
        stats = GenerationResult(
            generation=generation,
            pop_size=n,
            best_fitness=float(np.max(fitness)),
            mean_fitness=float(np.mean(fitness)),
            worst_fitness=float(np.min(fitness)),
            diversity=self.compute_diversity(population),
            n_unique=len(np.unique(population, axis=0)),
            pareto_size=len(self.archive),
            mutation_rate=self.config.mutation_prob,
            crossover_rate=self.config.crossover_prob,
        )
        return children, stats
