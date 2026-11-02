"""LLM-as-judge similarities. Model-agnostic: you supply the completion function.

Both judges share the same machinery:

* prompts are filled in a single pass, so text inside a response that happens to look like a
  placeholder (``{item_b}``) is never substituted;
* replies that cannot be parsed are retried (``retries``) before raising ``JudgeParseError``;
* identical items score 1.0 without a model call (``skip_identical``), and repeated
  comparisons of the same texts are served from a cache (``cache_size``), which matters
  because deterministic systems often repeat themselves exactly.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..prompts import DEFAULT_DIMENSIONS, SEMANTIC_EQUIVALENCE_PROMPT, Dimension, build_rubric_prompt
from ..types import SimilarityResult
from .base import Context, Similarity
from .text import default_normalize

CompleteFn = Callable[[str], str]
ACompleteFn = Callable[[str], Awaitable[str]]

_PLACEHOLDER = re.compile(r"\{(query|item_a|item_b)\}")


class JudgeParseError(ValueError):
    """The judge's reply contained no usable score."""


def render_prompt(template: str, *, query: Any = None, item_a: Any = "", item_b: Any = "") -> str:
    """Fill ``{query}``, ``{item_a}`` and ``{item_b}`` in one pass.

    Unlike chained ``str.replace`` (or ``str.format``), braces or placeholder-like text inside
    the values are left untouched, so a response containing ``{item_b}`` cannot pull the other
    response into its own slot.
    """
    values = {"query": str(query) if query not in (None, "") else "(not provided)",
              "item_a": str(item_a), "item_b": str(item_b)}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _json_objects(text: str):
    """Yield every JSON object embedded in ``text``, in order (tolerates prose and code fences)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        start = text.find("{", end)


# "1-5", "1 to 5", "out of 5", "/5" describe the scale, not the score: drop them before searching.
_SCALE_TEXT = re.compile(
    r"\(?\b\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\b\)?|\bout\s+of\s+\d+(?:\.\d+)?|/\s*\d+(?:\.\d+)?",
    re.I,
)
_NUMBER = r"(-?\d+(?:\.\d+)?)"


def parse_judge_output(text: str, scale: Tuple[float, float]) -> Tuple[float, Optional[str]]:
    """Extract (score, reason) from a judge reply.

    Tries, in order: a JSON object with a numeric ``score``; ``score: N`` / ``rating: N``;
    the last number in range. Scale descriptions such as ``1-5``, ``out of 5`` and ``/5`` are
    ignored so they are not mistaken for the score.
    """
    lo, hi = scale
    for obj in _json_objects(text):
        if "score" in obj:
            try:
                reason = obj.get("reason")
                return float(obj["score"]), None if reason is None else str(reason)
            except (TypeError, ValueError):
                pass
    cleaned = _SCALE_TEXT.sub(" ", text)
    m = re.search(r"(?i)\b(?:score|rating)\b[^\d\-]{0,12}" + _NUMBER, cleaned)
    if m:
        return float(m.group(1)), None
    nums = [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", cleaned)]
    nums = [x for x in nums if lo <= x <= hi]
    if nums:
        return nums[-1], None
    raise JudgeParseError(f"no score found in judge reply: {text[:200]!r}")


def parse_rubric_output(
    text: str, names: Sequence[str], scale: Tuple[float, float]
) -> Tuple[Dict[str, float], Optional[str]]:
    """Extract ({dimension: score}, reason) from a rubric reply (JSON with a ``scores`` object).

    Accepts the scores nested under ``"scores"`` or at the top level of the object. Values are
    clamped to ``scale``.
    """
    last_error: Optional[Exception] = None
    for obj in _json_objects(text):
        scores = obj.get("scores", obj)
        if not isinstance(scores, Mapping):
            continue
        try:
            out = {n: min(max(float(scores[n]), scale[0]), scale[1]) for n in names}
        except (KeyError, TypeError, ValueError) as e:
            last_error = e
            continue
        reason = obj.get("reason")
        return out, None if reason is None else str(reason)
    detail = f" ({last_error!r})" if last_error else ""
    raise JudgeParseError(f"missing or invalid dimension scores{detail}: {text[:200]!r}")


class _LRU:
    def __init__(self, size: int) -> None:
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key: str, value: Any) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


class _Judge(Similarity):
    """Shared plumbing for LLM judges: render, call, parse with retries, cache, symmetry."""

    def __init__(
        self,
        llm: Optional[CompleteFn],
        allm: Optional[ACompleteFn],
        *,
        prompt: str,
        scale: Tuple[float, float],
        symmetric: bool,
        retries: int,
        skip_identical: bool,
        cache_size: int,
    ) -> None:
        if llm is None and allm is None:
            raise ValueError("provide llm and/or allm")
        if scale[1] <= scale[0]:
            raise ValueError("scale must be (low, high) with high > low")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self._llm, self._allm = llm, allm
        self.prompt, self.scale, self.symmetric = prompt, scale, symmetric
        self.retries, self.skip_identical = retries, skip_identical
        self._cache = _LRU(cache_size)

    # subclasses define how a reply is parsed and how parsed replies become a result
    def _parse(self, reply: str) -> Any: ...

    def _finish(self, parsed: List[Any]) -> SimilarityResult: ...

    def _identical(self) -> SimilarityResult: ...

    def _render(self, a: Any, b: Any, context: Context) -> str:
        return render_prompt(self.prompt, query=(context or {}).get("query"), item_a=a, item_b=b)

    def _orders(self, a: Any, b: Any) -> List[Tuple[Any, Any]]:
        return [(a, b), (b, a)] if self.symmetric else [(a, b)]

    def _same(self, a: Any, b: Any) -> bool:
        return self.skip_identical and default_normalize(str(a)) == default_normalize(str(b))

    def _one(self, prompt: str) -> Any:
        cached = self._cache.get(prompt)
        if cached is not None:
            return cached
        error: Optional[JudgeParseError] = None
        for _ in range(self.retries + 1):
            try:
                parsed = self._parse(self._llm(prompt))  # type: ignore[misc]
            except JudgeParseError as e:
                error = e
                continue
            self._cache.put(prompt, parsed)
            return parsed
        raise error  # type: ignore[misc]

    async def _aone(self, prompt: str) -> Any:
        cached = self._cache.get(prompt)
        if cached is not None:
            return cached
        error: Optional[JudgeParseError] = None
        for _ in range(self.retries + 1):
            try:
                parsed = self._parse(await self._allm(prompt))  # type: ignore[misc]
            except JudgeParseError as e:
                error = e
                continue
            self._cache.put(prompt, parsed)
            return parsed
        raise error  # type: ignore[misc]

    def compute(self, a: Any, b: Any, *, context: Context = None) -> SimilarityResult:
        if self._same(a, b):
            return self._identical()
        if self._llm is None:
            raise RuntimeError("this judge only has an async llm; use aevaluate()/ascore()")
        return self._finish([self._one(self._render(x, y, context)) for x, y in self._orders(a, b)])

    async def acompute(self, a: Any, b: Any, *, context: Context = None) -> SimilarityResult:
        if self._same(a, b):
            return self._identical()
        if self._allm is None:
            return await super().acompute(a, b, context=context)  # type: ignore[return-value]
        prompts = [self._render(x, y, context) for x, y in self._orders(a, b)]
        return self._finish(list(await asyncio.gather(*(self._aone(p) for p in prompts))))


class LLMJudge(_Judge):
    """Semantic equivalence of two items, scored by an LLM on a 1-5 rubric.

    ``llm`` is any ``str -> str`` completion function (and/or ``allm``, its async twin), so
    this works with any provider or local model. The score is normalised to [0, 1]; the
    raw rating and the judge's reason are kept on the ``SimilarityResult``. Put the query
    both items answer in ``context={"query": ...}`` and the prompt will include it.

    ``symmetric=True`` judges both (a, b) and (b, a) and averages them, removing position
    bias at twice the cost. ``retries`` re-asks when a reply has no parsable score.
    ``skip_identical`` scores identical (whitespace/case-normalised) items 1.0 for free, and
    ``cache_size`` memoises recent judgements (0 disables).
    """

    name = "llm_judge"

    def __init__(
        self,
        llm: Optional[CompleteFn] = None,
        allm: Optional[ACompleteFn] = None,
        *,
        prompt: str = SEMANTIC_EQUIVALENCE_PROMPT,
        scale: Tuple[float, float] = (1.0, 5.0),
        symmetric: bool = False,
        retries: int = 2,
        skip_identical: bool = True,
        cache_size: int = 10000,
    ) -> None:
        super().__init__(llm, allm, prompt=prompt, scale=scale, symmetric=symmetric, retries=retries,
                         skip_identical=skip_identical, cache_size=cache_size)

    def _parse(self, reply: str) -> Tuple[float, Optional[str]]:
        raw, reason = parse_judge_output(reply, self.scale)
        return min(max(raw, self.scale[0]), self.scale[1]), reason

    def _finish(self, parsed: List[Any]) -> SimilarityResult:
        lo, hi = self.scale
        raw = sum(p[0] for p in parsed) / len(parsed)
        reason = next((p[1] for p in parsed if p[1]), None)
        return SimilarityResult(score=(raw - lo) / (hi - lo), raw=raw, reason=reason)

    def _identical(self) -> SimilarityResult:
        return SimilarityResult(1.0, raw=self.scale[1], reason="identical", metadata={"skipped": True})


class RubricJudge(_Judge):
    """Multi-dimension LLM judge: one call scores several aspects of agreement (1-5 each).

    Default dimensions: ``core_facts``, ``specifics``, ``supporting_detail``, ``stance``,
    ``format`` (see ``repeatability_evals.prompts.DEFAULT_DIMENSIONS``). The similarity score is
    their weighted mean, normalised to [0, 1]; the per-dimension scores are kept on the result
    (``metadata["dimensions"]``) and averaged per dimension in ``PairwiseConsistency`` details, so
    you can see *what* varies between runs (facts? format?) and not just *whether* they differ.

    Pass your own ``dimensions`` (any number, any weights) to adapt it, e.g. for agent traces.
    Weights are normalised to sum to 1. ``symmetric``, ``retries``, ``skip_identical`` and
    ``cache_size`` behave as in ``LLMJudge``.
    """

    name = "rubric_judge"

    def __init__(
        self,
        llm: Optional[CompleteFn] = None,
        allm: Optional[ACompleteFn] = None,
        *,
        dimensions: Sequence[Dimension] = DEFAULT_DIMENSIONS,
        prompt: Optional[str] = None,
        scale: Tuple[float, float] = (1.0, 5.0),
        symmetric: bool = False,
        retries: int = 2,
        skip_identical: bool = True,
        cache_size: int = 10000,
    ) -> None:
        if not dimensions or any(d.weight <= 0 for d in dimensions):
            raise ValueError("need at least one dimension, each with weight > 0")
        if len({d.name for d in dimensions}) != len(dimensions):
            raise ValueError("dimension names must be unique")
        self.dimensions = tuple(dimensions)
        super().__init__(llm, allm, prompt=prompt or build_rubric_prompt(self.dimensions), scale=scale,
                         symmetric=symmetric, retries=retries, skip_identical=skip_identical,
                         cache_size=cache_size)
        self._names = [d.name for d in self.dimensions]
        total = sum(d.weight for d in self.dimensions)
        self._weights = {d.name: d.weight / total for d in self.dimensions}

    def _parse(self, reply: str) -> Tuple[Dict[str, float], Optional[str]]:
        return parse_rubric_output(reply, self._names, self.scale)

    def _finish(self, parsed: List[Any]) -> SimilarityResult:
        lo, hi = self.scale
        raw_dims = {n: sum(p[0][n] for p in parsed) / len(parsed) for n in self._names}
        norm = {n: (v - lo) / (hi - lo) for n, v in raw_dims.items()}
        score = sum(self._weights[n] * norm[n] for n in self._names)
        raw = lo + score * (hi - lo)
        reason = next((p[1] for p in parsed if p[1]), None)
        return SimilarityResult(score, raw=raw, reason=reason, metadata={"dimensions": norm})

    def _identical(self) -> SimilarityResult:
        return SimilarityResult(1.0, raw=self.scale[1], reason="identical",
                                metadata={"dimensions": {n: 1.0 for n in self._names}, "skipped": True})
