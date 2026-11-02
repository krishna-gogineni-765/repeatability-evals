"""Set-overlap similarities over collections of hashable items (IDs, labels, URLs...)."""
from __future__ import annotations

from typing import Any, Hashable, Iterable, Set

from .base import Context, Similarity


def to_set(x: Iterable[Hashable]) -> Set[Hashable]:
    if isinstance(x, (str, bytes)):
        raise TypeError(
            "set similarities compare collections; got a bare string. "
            "Wrap it (e.g. [text]) or use a text similarity such as NgramJaccard."
        )
    return set(x)


class JaccardSimilarity(Similarity):
    """|A & B| / |A | B|. Two empty collections count as identical (1.0)."""

    name = "jaccard"

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        sa, sb = to_set(a), to_set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb)


class DiceSimilarity(Similarity):
    """2|A & B| / (|A| + |B|): like Jaccard but more forgiving of partial overlap."""

    name = "dice"

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        sa, sb = to_set(a), to_set(b)
        if not sa and not sb:
            return 1.0
        return 2 * len(sa & sb) / (len(sa) + len(sb))


class OverlapCoefficient(Similarity):
    """|A & B| / min(|A|, |B|): 1.0 when one collection is contained in the other."""

    name = "overlap_coefficient"

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        sa, sb = to_set(a), to_set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / min(len(sa), len(sb))
