import pytest

from repeatability_evals import (
    DiceSimilarity,
    EditSimilarity,
    EmbeddingCosine,
    ExactMatch,
    FunctionSimilarity,
    JaccardSimilarity,
    NgramJaccard,
    OverlapCoefficient,
    RankBiasedOverlap,
    RougeL,
    Similarity,
    SimilarityResult,
    as_similarity,
)
from repeatability_evals.similarity.text import ngram_set, tokenize


# ---- base contract --------------------------------------------------------
def test_call_returns_float_and_score_returns_result():
    sim = FunctionSimilarity(lambda a, b: 0.25)
    assert sim("x", "y") == 0.25
    assert isinstance(sim.score("x", "y"), SimilarityResult)


@pytest.mark.parametrize("bad", [1.5, -0.2, float("nan")])
def test_out_of_range_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        FunctionSimilarity(lambda a, b: bad)("x", "y")


def test_tiny_float_overshoot_is_clamped():
    assert FunctionSimilarity(lambda a, b: 1.0000000001)("x", "y") == 1.0


def test_as_similarity():
    assert isinstance(as_similarity(lambda a, b: 1.0), Similarity)
    j = JaccardSimilarity()
    assert as_similarity(j) is j
    with pytest.raises(TypeError):
        as_similarity(42)


def test_custom_subclass_can_return_rich_result():
    class S(Similarity):
        def compute(self, a, b, *, context=None):
            return SimilarityResult(0.5, raw=3, reason="because")
    r = S().score(1, 2)
    assert (r.score, r.raw, r.reason) == (0.5, 3, "because")


# ---- sets -----------------------------------------------------------------
def test_jaccard_dice_overlap():
    a, b = {1, 2, 3}, {2, 3, 4}
    assert JaccardSimilarity()(a, b) == pytest.approx(0.5)
    assert DiceSimilarity()(a, b) == pytest.approx(2 / 3)
    assert OverlapCoefficient()({1, 2}, {1, 2, 3, 4}) == 1.0
    for sim in (JaccardSimilarity(), DiceSimilarity(), OverlapCoefficient()):
        assert sim([], []) == 1.0
        assert sim([1], []) == 0.0
        assert sim([1, 2], [2, 1]) == 1.0          # order-insensitive
        assert sim([1, 1, 2], [1, 2]) == 1.0       # duplicates ignored


def test_set_similarities_refuse_bare_strings():
    with pytest.raises(TypeError):
        JaccardSimilarity()("abc", "abd")


# ---- text -----------------------------------------------------------------
def test_tokenize_and_ngram_set():
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert ngram_set("The quick brown fox", 2) == {
        ("the", "quick"), ("quick", "brown"), ("brown", "fox")}
    assert ngram_set("Hi there", 3) == {("hi", "there")}
    assert ngram_set("  ", 3) == set()
    with pytest.raises(ValueError):
        ngram_set("x", 0)


def test_exact_match_normalises_case_and_whitespace():
    assert ExactMatch()("Yes  it IS", "yes it is") == 1.0
    assert ExactMatch()("yes", "no") == 0.0
    assert ExactMatch(normalize=None)("A", "a") == 0.0


def test_ngram_jaccard_pools_chunks_and_survives_rechunking():
    text = "backup jobs run nightly and retention is thirty days for all gold tier policies"
    words = text.split()
    sim = NgramJaccard(n=2)
    rechunked = [" ".join(words[:7]), " ".join(words[6:])]   # one shared word at the seam
    assert sim([text], rechunked) > 0.9
    assert JaccardSimilarity()(["c1"], ["c2", "c3"]) == 0.0   # IDs would say "totally different"


def test_ngram_jaccard_edges():
    sim = NgramJaccard(n=2)
    assert sim("alpha beta gamma", "one two three") == 0.0
    assert sim([], []) == 1.0 and sim("", "") == 1.0
    assert sim("same words here", "same words here") == 1.0
    with pytest.raises(ValueError):
        NgramJaccard(n=0)


def test_rouge_l():
    sim = RougeL()
    assert sim("a b c d", "a b c d") == 1.0
    assert sim("a b c", "x y z") == 0.0
    assert sim("", "") == 1.0
    assert sim("the cat sat on the mat", "on the mat the cat sat") < 1.0   # order matters
    assert 0.0 < sim("the cat sat", "the cat ran") < 1.0


def test_edit_similarity():
    sim = EditSimilarity()
    assert sim("Hello  World", "hello world") == 1.0
    assert sim("abc", "xyz") == 0.0
    assert sim("", "") == 1.0
    assert 0.0 < sim("search,fetch,answer", "search,answer") < 1.0
    assert sim(["search", "fetch"], ["search", "fetch"]) == 1.0           # token lists
    assert sim(["search", "fetch"], ["fetch", "search"]) < 1.0


# ---- ranking --------------------------------------------------------------
def test_rank_biased_overlap_properties():
    sim = RankBiasedOverlap(p=0.9)
    assert sim(list("abcd"), list("abcd")) == pytest.approx(1.0)
    assert sim(list("abc"), list("xyz")) == 0.0
    assert sim([], []) == 1.0 and sim(list("a"), []) == 0.0
    top_swap = sim(list("abcd"), list("bacd"))
    bottom_swap = sim(list("abcd"), list("abdc"))
    assert 0.0 < top_swap < bottom_swap < 1.0           # top ranks matter more
    assert sim(list("abc"), list("cba")) < 1.0
    assert sim(["a", "a", "b"], ["a", "b"]) == pytest.approx(1.0)   # duplicates collapse
    with pytest.raises(ValueError):
        RankBiasedOverlap(p=1.0)


# ---- embeddings -----------------------------------------------------------
def test_embedding_cosine_with_cache_and_rescale():
    calls = []
    table = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [-1.0, 0.0], "z": [0.0, 0.0]}

    def embed(texts):
        calls.append(list(texts))
        return [table[t] for t in texts]

    clip = EmbeddingCosine(embed)
    assert clip("a", "a") == 1.0                       # identical text: no embed call
    assert clip("a", "b") == 0.0
    assert clip("a", "c") == 0.0                       # negative cosine clipped
    assert clip("a", "z") == 0.0                       # zero vector
    assert EmbeddingCosine(embed, rescale="affine")("a", "c") == 0.0
    assert EmbeddingCosine(embed, rescale="affine")("a", "b") == pytest.approx(0.5)
    n_before = len(calls)
    clip("a", "b")
    assert len(calls) == n_before                      # served from cache
    with pytest.raises(ValueError):
        EmbeddingCosine(embed, rescale="nope")


def test_embedding_cosine_rejects_wrong_vector_count():
    with pytest.raises(ValueError):
        EmbeddingCosine(lambda texts: [[1.0]])("x", "y")
