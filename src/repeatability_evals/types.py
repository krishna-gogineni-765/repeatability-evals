"""Result and data types shared across the package."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Pair = Tuple[int, int]


@dataclass(frozen=True)
class SimilarityResult:
    """Outcome of comparing two items. ``score`` is always in [0, 1] (1 = identical)."""

    score: float
    raw: Optional[float] = None  # the un-normalised value, e.g. a 1-5 judge rating
    reason: Optional[str] = None  # free-text rationale, e.g. from an LLM judge
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairScore:
    i: int
    j: int
    score: float
    raw: Optional[float] = None
    reason: Optional[str] = None
    dimensions: Optional[Dict[str, float]] = None  # per-rubric-dimension scores in [0, 1]


@dataclass
class ConsistencyResult:
    """Consistency of one group of N items (e.g. N runs of the same input)."""

    metric: str
    score: float  # headline value in [0, 1]; 1 = fully consistent
    min_score: float  # the weakest compared pair (the "worst-case" view)
    n_items: int
    n_pairs: int  # pairs actually compared (== number of similarity calls)
    pairs: List[PairScore] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
