"""The one abstraction everything else is built on: comparing two items."""
from __future__ import annotations

import asyncio
import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Union

from ..types import SimilarityResult

Context = Optional[Mapping[str, Any]]
_EPS = 1e-9


def coerce_result(value: Union[float, int, SimilarityResult], source: str) -> SimilarityResult:
    """Accept a bare number or a SimilarityResult; enforce the [0, 1] contract."""
    result = value if isinstance(value, SimilarityResult) else SimilarityResult(score=float(value))
    s = result.score
    if math.isnan(s) or s < -_EPS or s > 1 + _EPS:
        raise ValueError(f"{source} returned {s!r}; similarity scores must be in [0, 1]")
    if s != min(max(s, 0.0), 1.0):
        result = SimilarityResult(min(max(s, 0.0), 1.0), result.raw, result.reason, result.metadata)
    return result


class Similarity(ABC):
    """Compare two items and return how alike they are, in [0, 1] (1 = identical).

    Subclass and implement ``compute``. Return a float, or a ``SimilarityResult`` to
    attach a raw value or a rationale. Items can be anything (strings, sets of IDs,
    lists of chunks, tool-call traces); ``context`` carries side information such as the
    query both items answer. Override ``acompute`` if your similarity does async I/O, and
    ``prepare`` if it benefits from seeing all items of a group up front (e.g. to embed them
    in one batch).
    """

    name: str = "similarity"

    @abstractmethod
    def compute(
        self, a: Any, b: Any, *, context: Context = None
    ) -> Union[float, SimilarityResult]: ...

    async def acompute(
        self, a: Any, b: Any, *, context: Context = None
    ) -> Union[float, SimilarityResult]:
        return await asyncio.to_thread(self.compute, a, b, context=context)

    def prepare(self, items: Sequence[Any]) -> None:
        """Called by metrics with every item of a group before any comparison. No-op default."""
        return None

    async def aprepare(self, items: Sequence[Any]) -> None:
        await asyncio.to_thread(self.prepare, items)

    # public entry points -------------------------------------------------
    def score(self, a: Any, b: Any, *, context: Context = None) -> SimilarityResult:
        return coerce_result(self.compute(a, b, context=context), self.name)

    async def ascore(self, a: Any, b: Any, *, context: Context = None) -> SimilarityResult:
        return coerce_result(await self.acompute(a, b, context=context), self.name)

    def __call__(self, a: Any, b: Any, *, context: Context = None) -> float:
        return self.score(a, b, context=context).score


def _accepts_context(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "context" or p.kind is p.VAR_KEYWORD for p in params)


class FunctionSimilarity(Similarity):
    """Wrap a plain ``fn(a, b) -> float`` (and optionally an async twin) as a Similarity.

    If the function accepts a ``context`` keyword (or ``**kwargs``), it receives the context.
    """

    def __init__(
        self,
        fn: Callable[..., Union[float, SimilarityResult]],
        *,
        afn: Optional[Callable[..., Awaitable[Union[float, SimilarityResult]]]] = None,
        name: Optional[str] = None,
    ) -> None:
        self._fn, self._afn = fn, afn
        self._fn_ctx = _accepts_context(fn)
        self._afn_ctx = afn is not None and _accepts_context(afn)
        self.name = name or getattr(fn, "__name__", "function_similarity")

    def compute(self, a: Any, b: Any, *, context: Context = None):
        return self._fn(a, b, context=context) if self._fn_ctx else self._fn(a, b)

    async def acompute(self, a: Any, b: Any, *, context: Context = None):
        if self._afn is not None:
            return await (self._afn(a, b, context=context) if self._afn_ctx else self._afn(a, b))
        return await super().acompute(a, b, context=context)


def as_similarity(obj: Any) -> Similarity:
    if isinstance(obj, Similarity):
        return obj
    if callable(obj):
        return FunctionSimilarity(obj)
    raise TypeError(f"expected a Similarity or a callable, got {type(obj).__name__}")
