"""Run a metric over many groups (one group = the N runs of one input) and summarise."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .metrics import ON_ERROR, ConsistencyMetric, call_limit
from .stats import bootstrap_ci, mean
from .types import ConsistencyResult


@dataclass
class DatasetResult:
    metric: str
    score: float  # mean of per-group scores, over groups with at least 2 items
    n_groups: int  # groups that were scored
    ci: Optional[Tuple[float, float]] = None  # bootstrap CI of ``score``
    per_group: Dict[Hashable, ConsistencyResult] = field(default_factory=dict)
    degenerate: List[Hashable] = field(default_factory=list)  # < 2 items: nothing to compare
    failed: Dict[Hashable, str] = field(default_factory=dict)  # only with on_error="skip"

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
    metric: ConsistencyMetric, results: Dict[Hashable, ConsistencyResult], failed: Dict[Hashable, str],
    ci: bool, n_boot: int, seed: int,
) -> DatasetResult:
    scored = {k: r for k, r in results.items() if not r.details.get("degenerate")}
    degenerate = [k for k, r in results.items() if r.details.get("degenerate")]
    if not scored:
        raise ValueError(
            "no group could be scored: every group has fewer than 2 items"
            + (f" or failed ({len(failed)} failed)" if failed else "")
        )
    scores = [r.score for r in scored.values()]
    interval = bootstrap_ci(scores, n_boot=n_boot, seed=seed) if ci else None
    return DatasetResult(metric.name, mean(scores), len(scored), interval, scored, degenerate, failed)


def _check(on_error: str) -> None:
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}")


def evaluate_groups(
    groups: Mapping[Hashable, Sequence[Any]],
    metric: ConsistencyMetric,
    *,
    contexts: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    ci: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    on_error: str = "raise",
) -> DatasetResult:
    """``groups`` maps a group key (e.g. a question ID) to its N items; ``contexts`` optionally
    maps the same keys to per-group context such as ``{"query": "..."}`` for an LLM judge.

    Groups with fewer than 2 items cannot be compared; they are listed in ``degenerate`` and
    left out of the mean. With ``on_error="skip"`` a group whose evaluation raises is recorded
    in ``failed`` instead of aborting the run.
    """
    _check(on_error)
    results: Dict[Hashable, ConsistencyResult] = {}
    failed: Dict[Hashable, str] = {}
    for key, items in groups.items():
        try:
            results[key] = metric.evaluate(items, context=(contexts or {}).get(key))
        except Exception as e:
            if on_error == "raise":
                raise
            failed[key] = repr(e)
    return _summarise(metric, results, failed, ci, n_boot, seed)


async def aevaluate_groups(
    groups: Mapping[Hashable, Sequence[Any]],
    metric: ConsistencyMetric,
    *,
    contexts: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    concurrency: int = 4,
    max_calls: Optional[int] = None,
    ci: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    on_error: str = "raise",
) -> DatasetResult:
    """Async ``evaluate_groups``. ``concurrency`` is the number of groups in flight;
    ``max_calls`` (optional) caps simultaneous similarity calls across all of them, which is
    what you usually want for a rate-limited LLM judge."""
    _check(on_error)
    sem = asyncio.Semaphore(concurrency)

    async def one(key: Hashable, items: Sequence[Any]) -> Tuple[Hashable, Any]:
        async with sem:
            try:
                return key, await metric.aevaluate(items, context=(contexts or {}).get(key))
            except Exception as e:
                if on_error == "raise":
                    raise
                return key, e

    with call_limit(max_calls):
        out = await asyncio.gather(*(one(k, v) for k, v in groups.items()))
    results = {k: r for k, r in out if isinstance(r, ConsistencyResult)}
    failed = {k: repr(r) for k, r in out if not isinstance(r, ConsistencyResult)}
    return _summarise(metric, results, failed, ci, n_boot, seed)
