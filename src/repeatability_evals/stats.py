"""Small statistics helpers (standard library only)."""
from __future__ import annotations

import itertools
import math
import random
from typing import Optional, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() of an empty sequence")
    return sum(values) / len(values)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    if not values:
        raise ValueError("bootstrap_ci() of an empty sequence")
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def permutation_pvalue(
    within: Sequence[float], cross: Sequence[float], n_perm: int = 2000, seed: int = 0,
    min_resolution: int = 10,
) -> Optional[float]:
    """One-sided permutation p-value that ``cross`` scores are lower than ``within`` scores.

    Statistic: mean(within) - mean(cross). Pair labels are permuted (exactly when the number of
    relabellings is small, otherwise ``n_perm`` random ones). Ties count half (mid-p), which keeps
    the test usable for coarse scores such as exact match. Pair scores that share a run are not
    independent, so treat the result as a noise-aware screen rather than an exact test. Returns
    ``None`` when there are too few pairs for the test to ever reach ``1 / min_resolution``.
    """
    w, c = list(within), list(cross)
    if not w or not c:
        return None
    pooled, k = w + c, len(c)
    total = math.comb(len(pooled), k)
    if total < min_resolution:
        return None
    observed = sum(w) / len(w) - sum(c) / k
    whole = sum(pooled)

    def stat(cross_idx: Sequence[int]) -> float:
        cs = sum(pooled[i] for i in cross_idx)
        return (whole - cs) / (len(pooled) - k) - cs / k

    eps = 1e-12

    def weight(value: float) -> float:
        if value > observed + eps:
            return 1.0
        return 0.5 if value >= observed - eps else 0.0

    if total <= n_perm:
        hits = sum(weight(stat(idx)) for idx in itertools.combinations(range(len(pooled)), k))
        return hits / total
    rng = random.Random(seed)
    idx = list(range(len(pooled)))
    hits = sum(weight(stat(rng.sample(idx, k))) for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)
