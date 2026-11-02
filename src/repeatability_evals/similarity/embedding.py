"""Embedding-based semantic similarity. Bring your own embedding function: no dependencies."""
from __future__ import annotations

import asyncio
import math
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .base import Context, Similarity

EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]
AEmbedFn = Callable[[List[str]], Awaitable[Sequence[Sequence[float]]]]


class EmbeddingCosine(Similarity):
    """Cosine similarity of text embeddings.

    ``embed`` takes a list of strings and returns one vector each, which is the shape of
    ``SentenceTransformer.encode`` and of most embedding API clients, e.g.::

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sim = EmbeddingCosine(lambda texts: model.encode(texts).tolist())

    ``aembed`` is an optional async twin used by the async metrics. ``rescale="clip"`` maps
    negative cosines to 0 (the usual convention for text embeddings); ``"affine"`` maps
    [-1, 1] onto [0, 1]. Embeddings are cached per text, and metrics call ``prepare`` with all
    items of a group so they are embedded in one batch instead of one call per pair.
    """

    name = "embedding_cosine"

    def __init__(
        self,
        embed: Optional[EmbedFn] = None,
        *,
        aembed: Optional[AEmbedFn] = None,
        rescale: str = "clip",
        max_cache: int = 20000,
    ) -> None:
        if embed is None and aembed is None:
            raise ValueError("provide embed and/or aembed")
        if rescale not in ("clip", "affine"):
            raise ValueError("rescale must be 'clip' or 'affine'")
        self._embed, self._aembed = embed, aembed
        self._rescale, self._max_cache = rescale, max_cache
        self._cache: Dict[str, Sequence[float]] = {}
        self._lock = threading.Lock()

    # -- caching (thread-safe: async metrics run sync similarities in worker threads) --------
    def _lookup(self, texts: List[str]) -> Dict[str, Sequence[float]]:
        with self._lock:
            return {t: self._cache[t] for t in texts if t in self._cache}

    def _store(self, texts: List[str], vecs: Sequence[Sequence[float]]) -> Dict[str, Sequence[float]]:
        if len(vecs) != len(texts):
            raise ValueError("embed() must return one vector per input text")
        new = dict(zip(texts, vecs))
        with self._lock:
            if len(self._cache) + len(new) > self._max_cache:
                self._cache.clear()
            self._cache.update(new)
        return new

    def _vectors(self, texts: List[str]) -> Dict[str, Sequence[float]]:
        unique = list(dict.fromkeys(texts))
        found = self._lookup(unique)
        missing = [t for t in unique if t not in found]
        if missing:
            if self._embed is None:
                raise RuntimeError("this similarity only has an async embed; use the async metrics")
            found.update(self._store(missing, self._embed(missing)))
        return found

    async def _avectors(self, texts: List[str]) -> Dict[str, Sequence[float]]:
        if self._aembed is None:
            return await asyncio.to_thread(self._vectors, texts)
        unique = list(dict.fromkeys(texts))
        found = self._lookup(unique)
        missing = [t for t in unique if t not in found]
        if missing:
            found.update(self._store(missing, await self._aembed(missing)))
        return found

    def prepare(self, items: Sequence[Any]) -> None:
        if self._embed is not None:
            self._vectors([str(x) for x in items])

    async def aprepare(self, items: Sequence[Any]) -> None:
        await self._avectors([str(x) for x in items])

    # -- scoring -----------------------------------------------------------------------------
    def _cosine(self, va: Sequence[float], vb: Sequence[float]) -> float:
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0 or nb == 0:
            return 0.0
        cos = sum(x * y for x, y in zip(va, vb)) / (na * nb)
        cos = min(1.0, max(-1.0, cos))
        return max(0.0, cos) if self._rescale == "clip" else (cos + 1) / 2

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        ta, tb = str(a), str(b)
        if ta == tb:
            return 1.0
        vecs = self._vectors([ta, tb])
        return self._cosine(vecs[ta], vecs[tb])

    async def acompute(self, a: Any, b: Any, *, context: Context = None) -> float:
        ta, tb = str(a), str(b)
        if ta == tb:
            return 1.0
        if self._aembed is None:
            return await super().acompute(a, b, context=context)  # type: ignore[return-value]
        vecs = await self._avectors([ta, tb])
        return self._cosine(vecs[ta], vecs[tb])
