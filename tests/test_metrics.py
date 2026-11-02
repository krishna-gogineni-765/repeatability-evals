import asyncio

import pytest

from consistency_evals import (
    AllPairs,
    AnchorPairs,
    ClusterConsistency,
    ExactMatch,
    JaccardSimilarity,
    NgramJaccard,
    PairwiseConsistency,
    TotalAgreement,
)


# ---- PairwiseConsistency --------------------------------------------------
def test_identical_items_are_fully_consistent():
    r = PairwiseConsistency(ExactMatch()).evaluate(["x", "x", "x", "x"])
    assert r.score == 1.0 and r.min_score == 1.0 and r.n_items == 4


def test_one_outlier_all_pairs():
    r = PairwiseConsistency(ExactMatch(), sampler=AllPairs()).evaluate(["x", "x", "x", "y"])
    assert r.n_pairs == 6
    assert r.score == pytest.approx(0.5) and r.min_score == 0.0


def test_default_sampler_limits_similarity_calls_to_n():
    calls = []

    def sim(a, b):
        calls.append((a, b))
        return 1.0
    PairwiseConsistency(sim).evaluate(list("abcdefghij"))
    assert len(calls) == 10             # not 45


def test_retrieval_style_usage_with_preprocess():
    runs = [{"chunk_ids": ["a", "b"]}, {"chunk_ids": ["b", "a"]}, {"chunk_ids": ["a", "c"]}]
    m = PairwiseConsistency(JaccardSimilarity(), preprocess=lambda r: r["chunk_ids"],
                            sampler=AllPairs())
    assert m.evaluate(runs).score == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)


def test_agreement_threshold_and_custom_aggregate():
    m = PairwiseConsistency(lambda a, b: 1.0 if a == b else 0.2, sampler=AllPairs(),
                            agreement_threshold=0.9, aggregate=min)
    r = m.evaluate(["x", "x", "y"])
    assert r.score == 0.2 and r.details["agreement_rate"] == pytest.approx(1 / 3)


def test_anchor_sampler_compares_against_a_reference_run():
    r = PairwiseConsistency(ExactMatch(), sampler=AnchorPairs(0)).evaluate(["gold", "gold", "x"])
    assert r.n_pairs == 2 and r.score == pytest.approx(0.5)


def test_degenerate_groups_are_trivially_consistent():
    for items in ([], ["only"]):
        r = PairwiseConsistency(ExactMatch()).evaluate(items)
        assert r.score == 1.0 and r.n_pairs == 0 and r.details["degenerate"]


def test_async_matches_sync():
    async def run():
        m = PairwiseConsistency(NgramJaccard(2), concurrency=2)
        items = [["a b c d"], ["a b c e"], ["a b c d"], ["x y z w"]]
        return m.evaluate(items), await m.aevaluate(items)
    sync, asyn = asyncio.run(run())
    assert sync.score == pytest.approx(asyn.score) and sync.n_pairs == asyn.n_pairs


def test_result_to_dict_roundtrips_basic_fields():
    d = PairwiseConsistency(ExactMatch()).evaluate(["a", "a"]).to_dict()
    assert d["score"] == 1.0 and d["n_items"] == 2 and isinstance(d["pairs"], list)


# ---- ClusterConsistency ---------------------------------------------------
def test_cluster_majority_share_and_entropy():
    r = ClusterConsistency(ExactMatch(), threshold=1.0).evaluate(["x", "x", "x", "y"])
    assert r.score == 0.75
    assert r.details["cluster_sizes"] == [3, 1] and r.details["n_clusters"] == 2
    assert r.details["normalized_entropy"] == pytest.approx(0.4056, abs=1e-3)


def test_cluster_extremes():
    same = ClusterConsistency(ExactMatch(), threshold=1.0).evaluate(["x"] * 5)
    assert same.score == 1.0 and same.details["normalized_entropy"] == 0.0
    diff = ClusterConsistency(ExactMatch(), threshold=1.0).evaluate(list("abcd"))
    assert diff.score == 0.25 and diff.details["normalized_entropy"] == pytest.approx(1.0)


def test_cluster_threshold_controls_how_loose_a_match_is():
    items = ["a b c d", "a b c e", "w x y z"]
    loose = ClusterConsistency(NgramJaccard(2), threshold=0.4).evaluate(items)
    strict = ClusterConsistency(NgramJaccard(2), threshold=0.9).evaluate(items)
    assert loose.details["n_clusters"] == 2 and strict.details["n_clusters"] == 3
    with pytest.raises(ValueError):
        ClusterConsistency(ExactMatch(), threshold=1.5)


def test_cluster_async_matches_sync_and_bounds_comparisons():
    items = ["x", "y", "x", "z", "x", "y"]
    m = ClusterConsistency(ExactMatch(), threshold=1.0)
    sync, asyn = m.evaluate(items), asyncio.run(m.aevaluate(items))
    assert sync.details["clusters"] == asyn.details["clusters"]
    assert sync.n_pairs <= len(items) * sync.details["n_clusters"]


def test_cluster_degenerate():
    assert ClusterConsistency(ExactMatch()).evaluate(["a"]).score == 1.0


# ---- TotalAgreement -------------------------------------------------------
def test_total_agreement():
    m = TotalAgreement()
    assert m.evaluate(["Yes.", "yes.", " YES. "]).score == 1.0
    r = m.evaluate(["a", "a", "b"])
    assert r.score == 0.0 and r.details["n_distinct"] == 2
    assert TotalAgreement(normalize=str).evaluate(["a", "A"]).score == 0.0   # raw agreement
    assert m.evaluate(["only"]).score == 1.0
