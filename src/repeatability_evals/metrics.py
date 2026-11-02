"""Consistency metrics: turn a Similarity (or an equality test) into a score for N items.

A metric knows nothing about *what* the items are. Feed it the N final answers of N runs,
N sets of retrieved chunk IDs, N tool-call traces, N paraphrase-conditioned outputs... and
pick the Similarity that suits them.
"""
from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence

from .sampling import CycleSampler, PairSampler
from .similarity.base import Context, Similarity, as_similarity
from .similarity.text import default_normalize
from .stats import mean
from .types import ConsistencyResult, PairScore, SimilarityResult


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
    # Nothing to disagree with: zero or one item is trivially consistent.
    return ConsistencyResult(name, 1.0, 1.0, n_items, 0, [], {"degenerate": True})


def _pair_score(i: int, j: int, r: SimilarityResult) -> PairScore:
    return PairScore(i, j, r.score, r.raw, r.reason, r.metadata.get("dimensions"))


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
        concurrency: max simultaneous similarity calls in ``aevaluate``.
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
    ) -> None:
        self.similarity: Similarity = as_similarity(similarity)
        self.sampler = sampler or CycleSampler()
        self.preprocess = preprocess
        self.agreement_threshold = agreement_threshold
        self.aggregate = aggregate or mean
        self.name = name or f"pairwise_{self.similarity.name}"
        self.concurrency = concurrency

    def _prepare(self, items: Sequence[Any]) -> List[Any]:
        return [self.preprocess(x) for x in items] if self.preprocess else list(items)

    def _finish(self, n_items: int, pairs: List[PairScore]) -> ConsistencyResult:
        scores = [p.score for p in pairs]
        details = {"similarity": self.similarity.name, "sampler": type(self.sampler).__name__}
        if self.agreement_threshold is not None:
            details["agreement_rate"] = sum(s >= self.agreement_threshold for s in scores) / len(scores)
        dims = [p.dimensions for p in pairs if p.dimensions]
        if dims:  # rubric judges: mean score per dimension, so you can see *what* varies
            details["dimensions"] = {k: mean([d[k] for d in dims]) for k in dims[0]}
        return ConsistencyResult(
            self.name, self.aggregate(scores), min(scores), n_items, len(pairs), pairs, details
        )

    def evaluate(self, items: Sequence[Any], *, context: Context = None) -> ConsistencyResult:
        prepared = self._prepare(items)
        pairs = self.sampler.sample(len(prepared))
        if not pairs:
            return _degenerate(self.name, len(prepared))
        scored = [
            _pair_score(i, j, self.similarity.score(prepared[i], prepared[j], context=context))
            for i, j in pairs
        ]
        return self._finish(len(prepared), scored)

    async def aevaluate(
        self, items: Sequence[Any], *, context: Context = None
    ) -> ConsistencyResult:
        prepared = self._prepare(items)
        pairs = self.sampler.sample(len(prepared))
        if not pairs:
            return _degenerate(self.name, len(prepared))
        sem = asyncio.Semaphore(self.concurrency)

        async def one(i: int, j: int) -> PairScore:
            async with sem:
                r = await self.similarity.ascore(prepared[i], prepared[j], context=context)
            return _pair_score(i, j, r)

        scored = await asyncio.gather(*(one(i, j) for i, j in pairs))
        return self._finish(len(prepared), list(scored))


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
        sem = asyncio.Semaphore(self.concurrency)
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


class TotalAgreement(ConsistencyMetric):
    """1.0 if *all* items are identical after normalisation, else 0.0.

    The mean over a dataset is the total agreement rate (TAR@N of Atil et al., 2024).
    Pass ``normalize=str`` for raw-output agreement, or an answer extractor for parsed
    agreement.
    """

    name = "total_agreement"

    def __init__(self, normalize: Callable[[Any], Any] = default_normalize) -> None:
        self.normalize = normalize

    def evaluate(self, items: Sequence[Any], *, context: Context = None) -> ConsistencyResult:
        if len(items) < 2:
            return _degenerate(self.name, len(items))
        distinct = {self.normalize(x) for x in items}
        score = 1.0 if len(distinct) == 1 else 0.0
        return ConsistencyResult(
            self.name, score, score, len(items), 0, [], {"n_distinct": len(distinct)}
        )
