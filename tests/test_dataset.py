import asyncio

import pytest

from repeatability_evals import (
    AllPairs,
    ExactMatch,
    LLMJudge,
    PairwiseConsistency,
    aevaluate_groups,
    bootstrap_ci,
    evaluate_groups,
)


def metric():
    return PairwiseConsistency(ExactMatch(), sampler=AllPairs())


GROUPS = {"q1": ["a", "a", "a"], "q2": ["a", "b", "c"], "q3": ["x", "x", "y"]}


def test_dataset_mean_per_group_and_worst():
    res = evaluate_groups(GROUPS, metric())
    assert res.n_groups == 3
    assert res.score == pytest.approx((1.0 + 0.0 + 1 / 3) / 3)
    assert res.per_group["q1"].score == 1.0
    assert [k for k, _ in res.worst(2)] == ["q2", "q3"]
    assert {r["group"] for r in res.to_records()} == set(GROUPS)


def test_confidence_interval_brackets_the_mean_and_is_reproducible():
    res = evaluate_groups(GROUPS, metric(), seed=1)
    lo, hi = res.ci
    assert lo <= res.score <= hi
    assert res.ci == evaluate_groups(GROUPS, metric(), seed=1).ci
    assert evaluate_groups(GROUPS, metric(), ci=False).ci is None


def test_bootstrap_ci_edges():
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
    lo, hi = bootstrap_ci([0.5] * 10)
    assert lo == hi == 0.5
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_empty_dataset_is_an_error():
    with pytest.raises(ValueError):
        evaluate_groups({}, metric())


def test_per_group_context_reaches_the_judge():
    seen = []
    judge = LLMJudge(lambda p: seen.append(p) or '{"score": 5}')
    m = PairwiseConsistency(judge, sampler=AllPairs())
    evaluate_groups({"q1": ["a", "b"], "q2": ["c", "d"]}, m,
                    contexts={"q1": {"query": "FIRST?"}, "q2": {"query": "SECOND?"}}, ci=False)
    assert any("FIRST?" in p for p in seen) and any("SECOND?" in p for p in seen)


def test_async_dataset_matches_sync():
    sync = evaluate_groups(GROUPS, metric(), ci=False)
    asyn = asyncio.run(aevaluate_groups(GROUPS, metric(), ci=False, concurrency=2))
    assert sync.score == pytest.approx(asyn.score)
    assert {k: r.score for k, r in sync.per_group.items()} == \
           {k: r.score for k, r in asyn.per_group.items()}
