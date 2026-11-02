"""Embedding-based semantic similarity. Bring your own embedding function: no dependencies."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence

from .base import Context, Similarity

EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]


class EmbeddingCosine(Similarity):
    """Cosine similarity of text embeddings.

    ``embed`` takes a list of strings and returns one vector each, which is the shape of
    ``SentenceTransformer.encode`` and of most embedding API clients, e.g.::

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sim = EmbeddingCosine(lambda texts: model.encode(texts).tolist())

    ``rescale="clip"`` maps negative cosines to 0 (the usual convention for text
    embeddings); ``"affine"`` maps [-1, 1] onto [0, 1]. Embeddings are cached per text
    because the same item is compared against several others.
    """

    name = "embedding_cosine"

    def __init__(self, embed: EmbedFn, *, rescale: str = "clip", max_cache: int = 20000) -> None:
        if rescale not in ("clip", "affine"):
            raise ValueError("rescale must be 'clip' or 'affine'")
        self._embed, self._rescale, self._max_cache = embed, rescale, max_cache
        self._cache: Dict[str, Sequence[float]] = {}

    def _vectors(self, texts: List[str]) -> List[Sequence[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if len(self._cache) + len(missing) > self._max_cache:
                self._cache.clear()
            vecs = self._embed(missing)
            if len(vecs) != len(missing):
                raise ValueError("embed() must return one vector per input text")
            self._cache.update(zip(missing, vecs))
        return [self._cache[t] for t in texts]

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        ta, tb = str(a), str(b)
        if ta == tb:
            return 1.0
        va, vb = self._vectors([ta, tb])
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0 or nb == 0:
            return 0.0
        cos = sum(x * y for x, y in zip(va, vb)) / (na * nb)
        return max(0.0, cos) if self._rescale == "clip" else (cos + 1) / 2
