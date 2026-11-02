"""Similarity for *ordered* lists, where position matters (ranked retrieval, tool sequences)."""
from __future__ import annotations

from typing import Any, Hashable, List, Sequence

from .base import Context, Similarity


def _unique(seq: Sequence[Hashable]) -> List[Hashable]:
    return list(dict.fromkeys(seq))


class RankBiasedOverlap(Similarity):
    """Rank-biased overlap (Webber et al., 2010), extrapolated form on the common depth.

    Top ranks count more than lower ones; ``p`` sets how top-heavy (0.9 means roughly the
    top ~10 ranks carry most of the weight). Identical rankings score 1.0, disjoint 0.0.
    Lists are truncated to the shorter length; duplicates keep their first position.
    """

    name = "rank_biased_overlap"

    def __init__(self, p: float = 0.9) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        self.p = p

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        la, lb = _unique(list(a)), _unique(list(b))
        if not la and not lb:
            return 1.0
        k = min(len(la), len(lb))
        if k == 0:
            return 0.0
        p, seen_a, seen_b = self.p, set(), set()
        overlap, weighted = 0, 0.0
        for d in range(1, k + 1):
            x, y = la[d - 1], lb[d - 1]
            if x == y:
                overlap += 1
            else:
                overlap += (x in seen_b) + (y in seen_a)
            seen_a.add(x)
            seen_b.add(y)
            weighted += (overlap / d) * p**d
        return (overlap / k) * p**k + ((1 - p) / p) * weighted
