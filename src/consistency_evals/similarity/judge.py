"""LLM-as-judge semantic equivalence. Model-agnostic: you supply the completion function."""
from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Optional, Tuple

from ..prompts import SEMANTIC_EQUIVALENCE_PROMPT
from ..types import SimilarityResult
from .base import Context, Similarity

CompleteFn = Callable[[str], str]
ACompleteFn = Callable[[str], Awaitable[str]]


class JudgeParseError(ValueError):
    """The judge's reply contained no usable score."""


def parse_judge_output(text: str, scale: Tuple[float, float]) -> Tuple[float, Optional[str]]:
    """Extract (score, reason) from a judge reply: JSON first, then 'score: N', then last number."""
    lo, hi = scale
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"]), obj.get("reason")
        except (ValueError, TypeError):
            pass
    m = re.search(r"(?i)\b(?:score|rating)\b\D{0,12}(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), None
    nums = [float(x) for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", text)]
    nums = [x for x in nums if lo <= x <= hi]
    if nums:
        return nums[-1], None
    raise JudgeParseError(f"no score found in judge reply: {text[:200]!r}")


class LLMJudge(Similarity):
    """Semantic equivalence of two items, scored by an LLM on a 1-5 rubric.

    ``llm`` is any ``str -> str`` completion function (and/or ``allm``, its async twin), so
    this works with any provider or local model. The score is normalised to [0, 1]; the
    raw rating and the judge's reason are kept on the ``SimilarityResult``. Put the query
    both items answer in ``context={"query": ...}`` and the prompt will include it.

    ``symmetric=True`` judges both (a, b) and (b, a) and averages them, removing position
    bias at twice the cost.
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
    ) -> None:
        if llm is None and allm is None:
            raise ValueError("provide llm and/or allm")
        if scale[1] <= scale[0]:
            raise ValueError("scale must be (low, high) with high > low")
        self._llm, self._allm = llm, allm
        self.prompt, self.scale, self.symmetric = prompt, scale, symmetric

    def _render(self, a: Any, b: Any, context: Context) -> str:
        query = str((context or {}).get("query", "") or "(not provided)")
        return (
            self.prompt.replace("{query}", query)
            .replace("{item_a}", str(a))
            .replace("{item_b}", str(b))
        )

    def _parse(self, reply: str) -> Tuple[float, Optional[str]]:
        raw, reason = parse_judge_output(reply, self.scale)
        return min(max(raw, self.scale[0]), self.scale[1]), reason

    def _finish(self, parsed: list) -> SimilarityResult:
        lo, hi = self.scale
        raw = sum(p[0] for p in parsed) / len(parsed)
        reason = next((p[1] for p in parsed if p[1]), None)
        return SimilarityResult(score=(raw - lo) / (hi - lo), raw=raw, reason=reason)

    def compute(self, a: Any, b: Any, *, context: Context = None) -> SimilarityResult:
        if self._llm is None:
            raise RuntimeError("this judge only has an async llm; use aevaluate()/ascore()")
        orders = [(a, b), (b, a)] if self.symmetric else [(a, b)]
        return self._finish([self._parse(self._llm(self._render(x, y, context))) for x, y in orders])

    async def acompute(self, a: Any, b: Any, *, context: Context = None) -> SimilarityResult:
        if self._allm is None:
            return await super().acompute(a, b, context=context)  # type: ignore[return-value]
        orders = [(a, b), (b, a)] if self.symmetric else [(a, b)]
        parsed = [self._parse(await self._allm(self._render(x, y, context))) for x, y in orders]
        return self._finish(parsed)
