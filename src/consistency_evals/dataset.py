"""Run a metric over many groups (one group = the N runs of one input) and summarise."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .metrics import ConsistencyMetric
from .stats import bootstrap_ci, mean
from .types import ConsistencyResult


@dataclass
class DatasetResult:
    metric: str
    score: float  # mean of per-group scores
    n_groups: int
    ci: Optional[Tuple[float, float]] = None  # bootstrap CI of ``score``
    per_group: Dict[Hashable, ConsistencyResult] = field(default_factory=dict)

    def worst(self, k: int = 5) -> List[Tuple[Hashable, ConsistencyResult]]:
        """The ``k`` least consistent groups: where to look first when debugging."""
        return sorted(self.per_group.items(), key=lambda kv: kv[1].score)[:k]

    def to_records(self) -> List[Dict[str, Any]]:
        return [
            {"group": key, "score": r.score, "min_score": r.min_score,
             "n_items": r.n_items, "n_pairs": r.n_pairs, **{f"d_{k}": v for k, v in r.details.items()
                                                             if isinstance(v, (int, float, str, bool))}}
            for key, r in self.per_group.items()
        ]


def _summarise(
    metric: ConsistencyMetric, results: Dict[Hashable, ConsistencyResult],
    ci: bool, n_boot: int, seed: int,
) -> DatasetResult:
    if not results:
        raise ValueError("no groups to evaluate")
    scores = [r.score for r in results.values()]
    interval = bootstrap_ci(scores, n_boot=n_boot, seed=seed) if ci else None
    return DatasetResult(metric.name, mean(scores), len(results), interval, results)


def evaluate_groups(
    groups: Mapping[Hashable, Sequence[Any]],
    metric: ConsistencyMetric,
    *,
    contexts: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    ci: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
) -> DatasetResult:
    """``groups`` maps a group key (e.g. a question ID) to its N items; ``contexts`` optionally
    maps the same keys to per-group context such as ``{"query": "..."}`` for an LLM judge."""
    results = {
        key: metric.evaluate(items, context=(contexts or {}).get(key))
        for key, items in groups.items()
    }
    return _summarise(metric, results, ci, n_boot, seed)


async def aevaluate_groups(
    groups: Mapping[Hashable, Sequence[Any]],
    metric: ConsistencyMetric,
    *,
    contexts: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    concurrency: int = 4,
    ci: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
) -> DatasetResult:
    sem = asyncio.Semaphore(concurrency)

    async def one(key: Hashable, items: Sequence[Any]) -> Tuple[Hashable, ConsistencyResult]:
        async with sem:
            return key, await metric.aevaluate(items, context=(contexts or {}).get(key))

    pairs = await asyncio.gather(*(one(k, v) for k, v in groups.items()))
    return _summarise(metric, dict(pairs), ci, n_boot, seed)
