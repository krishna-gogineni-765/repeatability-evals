"""Small statistics helpers (standard library only)."""
from __future__ import annotations

import random
from typing import Sequence, Tuple


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
