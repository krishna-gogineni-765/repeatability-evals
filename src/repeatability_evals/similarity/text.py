"""Lexical text similarities (no model calls)."""
from __future__ import annotations

import difflib
import re
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union

from .base import Context, Similarity

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens (Unicode-aware, punctuation dropped)."""
    return _TOKEN_RE.findall(text.lower())


def default_normalize(text: str) -> str:
    """Case-fold and collapse whitespace."""
    return " ".join(text.casefold().split())


def ngram_set(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    """Word n-grams of ``text``. Text shorter than ``n`` yields one short gram."""
    if n < 1:
        raise ValueError("n must be >= 1")
    tokens = tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[k : k + n]) for k in range(len(tokens) - n + 1)}


class ExactMatch(Similarity):
    """1.0 if the normalised strings are equal, else 0.0."""

    name = "exact_match"

    def __init__(self, normalize: Optional[Callable[[str], str]] = default_normalize) -> None:
        self._normalize = normalize or (lambda s: s)

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        return 1.0 if self._normalize(str(a)) == self._normalize(str(b)) else 0.0


class NgramJaccard(Similarity):
    """Jaccard over word n-grams.

    Each item is a string or a sequence of strings (e.g. retrieved chunks); a sequence is
    pooled into one n-gram set. That makes it robust to re-chunking: when chunk IDs
    change but the underlying text does not, the score stays high.
    """

    name = "ngram_jaccard"

    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

    def _grams(self, item: Union[str, Sequence[str]]) -> Set[Tuple[str, ...]]:
        texts = [item] if isinstance(item, str) else list(item)
        grams: Set[Tuple[str, ...]] = set()
        for t in texts:
            grams |= ngram_set(t, self.n)
        return grams

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        ga, gb = self._grams(a), self._grams(b)
        if not ga and not gb:
            return 1.0
        return len(ga & gb) / len(ga | gb)


class RougeL(Similarity):
    """ROUGE-L F-measure: longest common token subsequence, so word order matters."""

    name = "rouge_l"

    @staticmethod
    def _lcs(x: List[str], y: List[str]) -> int:
        if not x or not y:
            return 0
        prev = [0] * (len(y) + 1)
        for xi in x:
            cur = [0]
            for j, yj in enumerate(y, 1):
                cur.append(prev[j - 1] + 1 if xi == yj else max(prev[j], cur[j - 1]))
            prev = cur
        return prev[-1]

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        ta, tb = tokenize(str(a)), tokenize(str(b))
        if not ta and not tb:
            return 1.0
        lcs = self._lcs(ta, tb)
        if lcs == 0:
            return 0.0
        p, r = lcs / len(tb), lcs / len(ta)
        return 2 * p * r / (p + r)


class EditSimilarity(Similarity):
    """Character-level similarity ratio (``difflib``); also works on lists of tokens."""

    name = "edit_similarity"

    def compute(self, a: Any, b: Any, *, context: Context = None) -> float:
        if isinstance(a, str):
            a, b = default_normalize(a), default_normalize(str(b))
        if len(a) == 0 and len(b) == 0:
            return 1.0
        return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
