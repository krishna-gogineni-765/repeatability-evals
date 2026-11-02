"""Choosing which pairs of items to compare.

Comparing all N-choose-2 pairs is quadratic; with an LLM judge that gets expensive. The
samplers here pick a subset deterministically (seeded) so results are reproducible.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol

from .types import Pair


class PairSampler(Protocol):
    def sample(self, n_items: int) -> List[Pair]:
        """Return distinct (i, j) pairs with i < j, drawn from range(n_items)."""
        ...


def _norm(a: int, b: int) -> Pair:
    return (a, b) if a < b else (b, a)


def cycle_pairs(n_items: int, budget: Optional[int] = None, seed: int = 0) -> List[Pair]:
    """A random Hamiltonian cycle over the items, topped up with random extra pairs.

    Every item appears in at least two sampled pairs when ``budget >= n_items``, so no
    item can hide from the comparison. ``budget`` defaults to ``n_items``. When the
    budget is smaller than the cycle, alternate cycle edges (a matching) are used first
    so as many items as possible are still covered. When it reaches N-choose-2 you get
    every pair.
    """
    if n_items < 2:
        return []
    all_pairs = list(itertools.combinations(range(n_items), 2))
    budget = n_items if budget is None else budget
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if budget >= len(all_pairs):
        return all_pairs

    rng = random.Random(seed)
    order = list(range(n_items))
    rng.shuffle(order)  # random cycle: no positional bias
    ring = list(dict.fromkeys(_norm(order[k], order[(k + 1) % n_items]) for k in range(n_items)))
    ring = ring[0::2] + ring[1::2]  # even edges form a matching -> best coverage if truncated
    chosen = ring[:budget]
    if len(chosen) < budget:
        taken = set(chosen)
        rest = [p for p in all_pairs if p not in taken]
        rng.shuffle(rest)
        chosen += rest[: budget - len(chosen)]
    return sorted(chosen)


@dataclass(frozen=True)
class AllPairs:
    """Exact: every pair. Cost grows quadratically."""

    def sample(self, n_items: int) -> List[Pair]:
        return list(itertools.combinations(range(n_items), 2))


@dataclass(frozen=True)
class CycleSampler:
    """Default. ``budget`` pairs (default: N) with coverage guaranteed; see ``cycle_pairs``."""

    budget: Optional[int] = None
    seed: int = 0

    def sample(self, n_items: int) -> List[Pair]:
        return cycle_pairs(n_items, self.budget, self.seed)


@dataclass(frozen=True)
class RandomPairs:
    """``budget`` distinct pairs drawn uniformly, with no coverage guarantee."""

    budget: int
    seed: int = 0

    def sample(self, n_items: int) -> List[Pair]:
        all_pairs = list(itertools.combinations(range(n_items), 2))
        if self.budget >= len(all_pairs):
            return all_pairs
        return sorted(random.Random(self.seed).sample(all_pairs, self.budget))


@dataclass(frozen=True)
class AnchorPairs:
    """Compare every item to one anchor (e.g. a reference or baseline run): N-1 pairs."""

    anchor: int = 0

    def sample(self, n_items: int) -> List[Pair]:
        if n_items < 2:
            return []
        if not 0 <= self.anchor < n_items:
            raise IndexError(f"anchor {self.anchor} out of range for {n_items} items")
        return [_norm(self.anchor, k) for k in range(n_items) if k != self.anchor]


@dataclass
class CrossSampler:
    """Pairs between two groups of runs, e.g. baseline runs vs. candidate runs.

    Treats ``A + B`` as one list of ``n_a + n_b`` items and returns only (a, b) pairs. The
    default budget is ``max(n_a, n_b)``: every run on both sides is compared at least once.
    Use ``budget=None`` with small groups for all ``n_a * n_b`` pairs via ``budget=n_a * n_b``.
    """

    n_a: int
    budget: Optional[int] = None
    seed: int = 0

    def sample(self, n_items: int) -> List[Pair]:
        n_a, n_b = self.n_a, n_items - self.n_a
        if n_a < 1 or n_b < 1:
            return []
        total = n_a * n_b
        budget = max(n_a, n_b) if self.budget is None else self.budget
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if budget >= total:
            return [(i, n_a + j) for i in range(n_a) for j in range(n_b)]
        rng = random.Random(self.seed)
        a, b = list(range(n_a)), list(range(n_b))
        rng.shuffle(a)
        rng.shuffle(b)
        span = max(n_a, n_b)  # span <= lcm(n_a, n_b), so these pairs are distinct
        chosen = [(a[k % n_a], b[k % n_b]) for k in range(span)]
        if budget < span:
            chosen = chosen[:budget]
        else:
            rest = [(i, j) for i in range(n_a) for j in range(n_b) if (i, j) not in set(chosen)]
            rng.shuffle(rest)
            chosen += rest[: budget - span]
        return [(i, n_a + j) for i, j in chosen]
