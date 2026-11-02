"""Consistency metrics: turn a Similarity (or an equality test) into a score for N items.

A metric knows nothing about *what* the items are. Feed it the N final answers of N runs,
N sets of retrieved chunk IDs, N tool-call traces, N paraphrase-conditioned outputs... and
pick the Similarity that suits them.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

from .sampling import CycleSampler, PairSampler
from .similarity.base import Context, Similarity, as_similarity
from .similarity.text import default_normalize
from .stats import mean
from .types import ConsistencyResult, PairScore, SimilarityResult

ON_ERROR = ("raise", "skip")
_MAX_ERRORS_KEPT = 5

# A semaphore shared by every similarity call in the current async task tree. Set by
# ``call_limit`` (used by ``acompare_versions`` / ``aevaluate_groups``) so a global cap holds
# even when many groups and metrics run at once. Without it each ``aevaluate`` uses its own.
_CALL_LIMIT: contextvars.ContextVar[Optional[asyncio.Semaphore]] = contextvars.ContextVar(
    "repeatability_evals_call_limit", default=None
)


@contextlib.contextmanager
def call_limit(max_calls: Optional[int]) -> Iterator[None]:
    """Cap concurrent similarity calls across everything awaited inside this block.

    Must be entered inside a running event loop::

        async def main():
            with call_limit(8):
                await asyncio.gather(m.aevaluate(a), m.aevaluate(b))
    """
    if max_calls is None:
        yield
        return
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    token = _CALL_LIMIT.set(asyncio.Semaphore(max_calls))
    try:
        yield
    finally:
        _CALL_LIMIT.reset(token)


class AllComparisonsFailed(RuntimeError):
    """Every comparison in a group raised (only with ``on_error="skip"``)."""


class ConsistencyMetric(ABC):
    """Score how consistent a group of N items is. Higher is more consistent (1.0 = fully)."""

    name: str = "consistency"

    @abstractmethod
    def evaluate(self, items: Sequence[Any], *, context: Context = None) -> ConsistencyResult: ...

    async def aevaluate(
        self, items: Sequence[Any], *, context: Context = None
    ) -> ConsistencyResult:
        return await asyncio.to_thread(self.evaluate, items, context=context)


def _degenerate(name: str, n_items: int) -> ConsistencyResult:
    # Nothing to disagree with: zero or one item is trivially consistent. Dataset-level
    # summaries exclude these groups from their means (see ``details["degenerate"]``).
    return ConsistencyResult(name, 1.0, 1.0, n_items, 0, [], {"degenerate": True})


def _pair_score(i: int, j: int, r: SimilarityResult) -> PairScore:
    return PairScore(i, j, r.score, r.raw, r.reason, r.metadata.get("dimensions"))


def _check_on_error(on_error: str) -> str:
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}")
    return on_error


def _semaphore(concurrency: int) -> asyncio.Semaphore:
    return _CALL_LIMIT.get() or asyncio.Semaphore(concurrency)


class PairwiseConsistency(ConsistencyMetric):
    """Average similarity over a sampled set of item pairs.

    Args:
        similarity: a ``Similarity`` (or any ``fn(a, b) -> float``).
        sampler: which pairs to compare. Default ``CycleSampler()``: N pairs, every item
            covered, deterministic. Use ``AllPairs()`` for the exact all-pairs value.
        preprocess: applied to each item before comparing, e.g. ``lambda r: r["chunk_ids"]``.
        agreement_threshold: if set, ``details["agreement_rate"]`` reports the share of
            compared pairs scoring at least this (on the normalised [0, 1] scale).
        aggregate: how to combine pair scores into ``score`` (default: mean).
        concurrency: max simultaneous similarity calls in ``aevaluate`` (unless a wider
            ``call_limit`` is active).
        on_error: ``"raise"`` (default) propagates the first failing comparison.
            ``"skip"`` drops failing pairs, scores the rest and reports
            ``details["n_failed"]`` and ``details["errors"]``; if every pair fails it raises
            ``AllComparisonsFailed``. Useful for long LLM-judge runs where one bad reply or
            timeout should not abort the whole evaluation.
    """

    def __init__(
        self,
        similarity: Any,
        *,
        sampler: Optional[PairSampler] = None,
        preprocess: Optional[Callable[[Any], Any]] = None,
        agreement_threshold: Optional[float] = None,
        aggregate: Optional[Callable[[Sequence[float]], float]] = None,
        name: Optional[str] = None,
        concurrency: int = 8,
        on_error: str = "raise",
    ) -> None:
        self.similarity: Similarity = as_similarity(similarity)
        self.sampler = sampler or CycleSampler()
        self.preprocess = preprocess
        self.agreement_threshold = agreement_threshold
        self.aggregate = aggregate or mean
        self.name = name or f"pairwise_{self.similarity.name}"
        self.concurrency = concurrency
        self.on_error = _check_on_error(on_error)

    def _prepare(self, items: Sequence[Any]) -> List[Any]:
        return [self.preprocess(x) for x in items] if self.preprocess else list(items)

    def _finish(
        self, n_items: int, pairs: List[PairScore], errors: List[Tuple[int, int, BaseException]]
    ) -> ConsistencyResult:
        if not pairs:
            raise AllComparisonsFailed(
                f"all {len(errors)} comparisons failed; first error: {errors[0][2]!r}"
            ) from errors[0][2]
        scores = [p.score for p in pairs]
        details = {"similarity": self.similarity.name, "sampler": type(self.sampler).__name__}
        if self.agreement_threshold is not None:
            details["agreement_rate"] = sum(s >= self.agreement_threshold for s in scores) / len(scores)
        dims = [p.dimensions for p in pairs if p.dimensions]
        if dims:  # rubric judges: mean score per dimension, so you can see *what* varies
            details["dimensions"] = {k: mean([d[k] for d in dims]) for k in dims[0]}
        if errors:
            details["n_failed"] = len(errors)
            details["errors"] = [f"({i}, {j}): {e!r}" for i, j, e in errors[:_MAX_ERRORS_KEPT]]
        return ConsistencyResult(
            self.name, self.aggregate(scores), min(scores), n_items, len(pairs), pairs, details
        )

    def evaluate(self, items: Sequence[Any], *, context: Context = None) -> ConsistencyResult:
        prepared = self._prepare(items)
        pairs = self.sampler.sample(len(prepared))
        if not pairs:
            return _degenerate(self.name, len(prepared))
        self.similarity.prepare(prepared)
        scored: List[PairScore] = []
        errors: List[Tuple[int, int, BaseException]] = []
        for i, j in pairs:
            try:
                r = self.similarity.score(prepared[i], prepared[j], context=context)
            except Exception as e:
                if self.on_error == "raise":
                    raise
                errors.append((i, j, e))
                continue
            scored.append(_pair_score(i, j, r))
        return self._finish(len(prepared), scored, errors)

    async def aevaluate(
        self, items: Sequence[Any], *, context: Context = None
    ) -> ConsistencyResult:
        prepared = self._prepare(items)
        pairs = self.sampler.sample(len(prepared))
        if not pairs:
            return _degenerate(self.name, len(prepared))
        await self.similarity.aprepare(prepared)
        sem = _semaphore(self.concurrency)

        async def one(i: int, j: int) -> Any:
            async with sem:
                try:
                    r = await self.similarity.ascore(prepared[i], prepared[j], context=context)
                except Exception as e:
                    if self.on_error == "raise":
                        raise
                    return (i, j, e)
            return _pair_score(i, j, r)

        out = await asyncio.gather(*(one(i, j) for i, j in pairs))
        scored = [x for x in out if isinstance(x, PairScore)]
        errors = [x for x in out if not isinstance(x, PairScore)]
        return self._finish(len(prepared), scored, errors)


class ClusterConsistency(ConsistencyMetric):
    """Group items into 'same answer' clusters; the score is the largest cluster's share.

    Greedy: each item joins the first cluster whose first member it matches
    (``similarity >= threshold``), else starts a new one. Order-dependent by construction;
    it never compares more than items x clusters pairs. ``details`` reports the cluster
    sizes and the normalised entropy of the partition (the discrete form of semantic
    entropy: 0 = one cluster, 1 = every item alone).
    """

    def __init__(
        self, similarity: Any, *, threshold: float = 0.75, name: Optional[str] = None,
        preprocess: Optional[Callable[[Any], Any]] = None, concurrency: int = 8,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.similarity: Similarity = as_similarity(similarity)
        self.threshold, self.preprocess, self.concurrency = threshold, preprocess, concurrency
        self.name = name or f"cluster_{self.similarity.name}"

    def _prepare(self, items: Sequence[Any]) -> List[Any]:
        return [self.preprocess(x) for x in items] if self.preprocess else list(items)

    def _result(self, n: int, members: List[List[int]], pairs: List[PairScore]) -> ConsistencyResult:
        sizes = sorted((len(m) for m in members), reverse=True)
        probs = [s / n for s in sizes]
        entropy = -sum(p * math.log(p) for p in probs)
        norm = entropy / math.log(n) if n > 1 else 0.0
        details = {
            "similarity": self.similarity.name, "threshold": self.threshold,
            "cluster_sizes": sizes, "n_clusters": len(sizes), "normalized_entropy": norm,
            "clusters": members,
        }
        scores = [p.score for p in pairs]
        return ConsistencyResult(
            self.name, sizes[0] / n, min(scores) if scores else 1.0, n, len(pairs), pairs, details
        )

    def evaluate(self, items: Sequence[Any], *, context: Context = None) -> ConsistencyResult:
        prepared = self._prepare(items)
        if len(prepared) < 2:
            return _degenerate(self.name, len(prepared))
        self.similarity.prepare(prepared)
        members: List[List[int]] = []
        pairs: List[PairScore] = []
        for idx, item in enumerate(prepared):
            for cluster in members:
                r = self.similarity.score(prepared[cluster[0]], item, context=context)
                pairs.append(_pair_score(cluster[0], idx, r))
                if r.score >= self.threshold:
                    cluster.append(idx)
                    break
            else:
                members.append([idx])
        return self._result(len(prepared), members, pairs)

    async def aevaluate(
        self, items: Sequence[Any], *, context: Context = None
    ) -> ConsistencyResult:
        prepared = self._prepare(items)
        if len(prepared) < 2:
            return _degenerate(self.name, len(prepared))
        await self.similarity.aprepare(prepared)
        sem = _semaphore(self.concurrency)
        members: List[List[int]] = []
        pairs: List[PairScore] = []

        async def cmp(rep: int, idx: int) -> PairScore:
            async with sem:
                r = await self.similarity.ascore(prepared[rep], prepared[idx], context=context)
            return _pair_score(rep, idx, r)

        for idx in range(len(prepared)):
            # compare against every current cluster at once, then take the first match
            results = await asyncio.gather(*(cmp(c[0], idx) for c in members))
            pairs.extend(results)
            hit = next((k for k, p in enumerate(results) if p.score >= self.threshold), None)
            if hit is None:
                members.append([idx])
            else:
                members[hit].append(idx)
        return self._result(len(prepared), members, pairs)


def _normalize_any(x: Any) -> Any:
    """Default TotalAgreement key: normalised text for strings, canonical JSON otherwise."""
    if isinstance(x, str):
        return default_normalize(x)
    return json.dumps(x, sort_keys=True, default=str)


class TotalAgreement(ConsistencyMetric):
    """1.0 if *all* items are identical after normalisation, else 0.0.

    The mean over a dataset is the total agreement rate (TAR@N of Atil et al., 2024).
    Pass ``normalize=str`` for raw-output agreement, or an answer extractor for parsed
    agreement.
    """

    name = "total_agreement"

    def __init__(self, normalize: Optional[Callable[[Any], Any]] = None) -> None:
        self.normalize = normalize or _normalize_any

    def evaluate(self, items: Sequence[Any], *, context: Context = None) -> ConsistencyResult:
        if len(items) < 2:
            return _degenerate(self.name, len(items))
        distinct = {self.normalize(x) for x in items}
        score = 1.0 if len(distinct) == 1 else 0.0
        return ConsistencyResult(
            self.name, score, score, len(items), 0, [], {"n_distinct": len(distinct)}
        )
