"""Regression tests for the 0.3.0 robustness, cost and correctness fixes."""
import asyncio
import json

import pytest

from repeatability_evals import (
    AllComparisonsFailed,
    AllPairs,
    EmbeddingCosine,
    ExactMatch,
    FunctionSimilarity,
    JudgeParseError,
    LLMJudge,
    PairwiseConsistency,
    RubricJudge,
    TotalAgreement,
    acompare_versions,
    aevaluate_groups,
    call_limit,
    compare_versions,
    evaluate_groups,
    render_prompt,
)
from repeatability_evals.cli import main
from repeatability_evals.similarity.judge import parse_judge_output, parse_rubric_output

SCALE = (1.0, 5.0)
DIMS = ["core_facts", "specifics", "supporting_detail", "stance", "format"]


# ---- judge reply parsing --------------------------------------------------
@pytest.mark.parametrize("reply,score", [
    ("Score (1-5): 4", 4.0),                       # the scale is not the score
    ("I'd give it a 2 out of 5", 2.0),
    ("Rating: 3/5", 3.0),
    ("On a scale of 1 to 5 this is a 4", 4.0),
    ('Let me think {about it}. {"score": 4, "reason": "close"}', 4.0),   # prose braces before JSON
    ('```json\n{"reason": "x", "score": "3"}\n```', 3.0),
])
def test_parse_judge_output_edge_cases(reply, score):
    assert parse_judge_output(reply, SCALE)[0] == score


def test_parse_keeps_reason_after_prose_braces():
    assert parse_judge_output('{x} {"score": 4, "reason": "close"}', SCALE) == (4.0, "close")


def test_rubric_parse_accepts_top_level_scores_and_skips_bad_objects():
    reply = '{"note": 1} ' + json.dumps({d: 4 for d in DIMS} | {"reason": "r"})
    scores, reason = parse_rubric_output(reply, DIMS, SCALE)
    assert set(scores.values()) == {4.0} and reason == "r"


# ---- prompt rendering -----------------------------------------------------
def test_render_prompt_is_single_pass():
    out = render_prompt("Q={query} A={item_a} B={item_b}", query="q {item_a}", item_a="has {item_b}",
                        item_b="SECRET")
    assert out == "Q=q {item_a} A=has {item_b} B=SECRET"
    assert out.count("SECRET") == 1


def test_judge_prompt_cannot_be_hijacked_by_placeholders_in_responses():
    seen = []
    LLMJudge(lambda p: seen.append(p) or '{"score": 5}').score("use {item_b}", "SECRET")
    assert seen[0].count("SECRET") == 1
    assert "<response_a>" in seen[0] and "<response_b>" in seen[0]


# ---- retries, identical skip, cache ---------------------------------------
def test_retries_unparsable_replies_then_succeeds():
    replies = iter(["no idea", "still no", '{"score": 5}'])
    assert LLMJudge(lambda p: next(replies), retries=2)("a", "b") == 1.0


def test_retries_exhausted_raises():
    with pytest.raises(JudgeParseError):
        LLMJudge(lambda p: "nope", retries=1)("a", "b")


def test_identical_items_skip_the_model():
    calls = []
    judge = LLMJudge(lambda p: calls.append(p) or '{"score": 1}')
    assert judge("Same answer.", "same  answer.") == 1.0 and calls == []
    rubric = RubricJudge(lambda p: calls.append(p) or "{}")
    r = rubric.score("x", "x")
    assert r.score == 1.0 and r.metadata["dimensions"]["format"] == 1.0 and calls == []
    assert LLMJudge(lambda p: '{"score": 1}', skip_identical=False)("x", "x") == 0.0


def test_repeated_comparisons_are_cached():
    calls = []
    judge = LLMJudge(lambda p: calls.append(p) or '{"score": 4}')
    judge("a", "b")
    judge("a", "b")
    judge("a", "b", context={"query": "other"})       # different prompt -> new call
    assert len(calls) == 2
    nocache = LLMJudge(lambda p: calls.append(p) or '{"score": 4}', cache_size=0)
    nocache("a", "b")
    nocache("a", "b")
    assert len(calls) == 4


def test_symmetric_async_runs_both_orders_concurrently():
    active = peak = 0

    async def allm(p):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return '{"score": 5}'
    asyncio.run(LLMJudge(allm=allm, symmetric=True).ascore("a", "b"))
    assert peak == 2


# ---- on_error="skip" --------------------------------------------------------
def flaky(a, b):
    if "bad" in (a, b):
        raise TimeoutError("judge timed out")
    return 1.0


def test_metric_skips_failed_pairs_and_reports_them():
    m = PairwiseConsistency(flaky, sampler=AllPairs(), on_error="skip")
    r = m.evaluate(["x", "x", "bad"])
    assert r.n_pairs == 1 and r.score == 1.0
    assert r.details["n_failed"] == 2 and "TimeoutError" in r.details["errors"][0]
    with pytest.raises(TimeoutError):
        PairwiseConsistency(flaky, sampler=AllPairs()).evaluate(["x", "bad"])
    with pytest.raises(AllComparisonsFailed):
        m.evaluate(["bad", "bad"])
    with pytest.raises(ValueError):
        PairwiseConsistency(flaky, on_error="ignore")


def test_async_metric_skips_failed_pairs():
    m = PairwiseConsistency(flaky, sampler=AllPairs(), on_error="skip")
    r = asyncio.run(m.aevaluate(["x", "x", "bad"]))
    assert r.n_pairs == 1 and r.details["n_failed"] == 2


def test_dataset_records_failed_groups():
    m = PairwiseConsistency(flaky, sampler=AllPairs(), on_error="skip")
    res = evaluate_groups({"ok": ["x", "x"], "dead": ["bad", "bad"]}, m, ci=False, on_error="skip")
    assert res.n_groups == 1 and "dead" in res.failed
    ares = asyncio.run(aevaluate_groups({"ok": ["x", "x"], "dead": ["bad", "bad"]}, m, ci=False,
                                        on_error="skip"))
    assert ares.n_groups == 1 and "dead" in ares.failed


def test_compare_records_failed_inputs():
    base = {"ok": ["x", "x"], "dead": ["bad", "bad"]}
    r = compare_versions(base, base, flaky, on_error="skip")
    assert r.n_inputs == 1 and "dead" in r.failed_inputs
    ra = asyncio.run(acompare_versions(base, base, flaky, on_error="skip"))
    assert ra.n_inputs == 1 and "dead" in ra.failed_inputs


# ---- dataset means ignore groups that cannot be compared ------------------
def test_single_run_groups_do_not_inflate_the_dataset_mean():
    res = evaluate_groups({"a": ["x", "y"], "b": ["only"], "c": []}, PairwiseConsistency(ExactMatch()),
                          ci=False)
    assert res.score == 0.0 and res.n_groups == 1 and sorted(res.degenerate) == ["b", "c"]
    with pytest.raises(ValueError, match="fewer than 2"):
        evaluate_groups({"b": ["only"]}, PairwiseConsistency(ExactMatch()))


# ---- concurrency caps are global ------------------------------------------
def tracking_judge():
    state = {"active": 0, "peak": 0}

    async def allm(p):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.005)
        state["active"] -= 1
        return '{"score": 5}'
    return LLMJudge(allm=allm, cache_size=0), state


def test_acompare_concurrency_is_a_global_cap():
    judge, state = tracking_judge()
    base = {k: [f"a{k}{i}" for i in range(4)] for k in range(15)}
    cand = {k: [f"b{k}{i}" for i in range(4)] for k in range(15)}
    asyncio.run(acompare_versions(base, cand, judge, concurrency=3))
    assert state["peak"] == 3


def test_aevaluate_groups_max_calls_and_call_limit():
    judge, state = tracking_judge()
    groups = {k: [f"r{k}{i}" for i in range(5)] for k in range(10)}
    asyncio.run(aevaluate_groups(groups, PairwiseConsistency(judge), concurrency=10, max_calls=2,
                                 ci=False))
    assert state["peak"] == 2

    async def manual():
        with call_limit(1):
            await asyncio.gather(*(PairwiseConsistency(judge).aevaluate(v) for v in groups.values()))
    state["peak"] = 0
    asyncio.run(manual())
    assert state["peak"] == 1


# ---- embeddings ------------------------------------------------------------
def test_embeddings_are_batched_once_per_group():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]
    PairwiseConsistency(EmbeddingCosine(embed), sampler=AllPairs()).evaluate(["a", "bb", "ccc", "dddd"])
    assert len(calls) == 1 and sorted(calls[0]) == ["a", "bb", "ccc", "dddd"]


def test_async_embed_is_used_by_async_metrics():
    calls = []

    async def aembed(texts):
        calls.append(list(texts))
        return [[1.0, float(len(t))] for t in texts]
    sim = EmbeddingCosine(aembed=aembed)
    r = asyncio.run(PairwiseConsistency(sim, sampler=AllPairs()).aevaluate(["a", "bb", "ccc"]))
    assert len(calls) == 1 and 0 < r.score <= 1
    with pytest.raises(RuntimeError):
        EmbeddingCosine(aembed=aembed)("x", "y")
    with pytest.raises(ValueError):
        EmbeddingCosine()


def test_embedding_cache_is_safe_under_threads():
    sim = EmbeddingCosine(lambda texts: [[1.0, float(len(t))] for t in texts], max_cache=3)
    items = [f"text {i}" * (i % 7 + 1) for i in range(60)]
    m = PairwiseConsistency(sim, concurrency=16)
    r = asyncio.run(m.aevaluate(items))     # sync embed runs in worker threads; tiny cache churns
    assert r.n_pairs == 60


# ---- misc -------------------------------------------------------------------
def test_function_similarity_receives_context_when_it_asks_for_it():
    seen = []
    sim = FunctionSimilarity(lambda a, b, context=None: seen.append(context) or 1.0)
    PairwiseConsistency(sim).evaluate(["x", "y"], context={"query": "q"})
    assert seen[0] == {"query": "q"}
    assert FunctionSimilarity(lambda a, b: 0.5)("x", "y", context={"q": 1}) == 0.5


def test_total_agreement_handles_unhashable_items():
    assert TotalAgreement().evaluate([{"a": [1]}, {"a": [1]}]).score == 1.0
    assert TotalAgreement().evaluate([["x"], ["y"]]).score == 0.0


def test_html_report_is_structured_and_escaped():
    r = compare_versions({"q": ["<script>x</script>"] * 3}, {"q": ["other"] * 3}, ExactMatch())
    html = r.to_html()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<table>" in html and "<h1>" in html and "class='changed'" in html


def test_cli_jsonl_input(tmp_path, capsys):
    rows = [{"id": f"q{i}", "output": "retention is thirty days", "query": "how long?"}
            for i in range(5) for _ in range(3)]
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert main(["compare", str(p), str(p)]) == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"output": "no id"}\n')
    with pytest.raises(SystemExit, match="needs"):
        main(["compare", str(bad), str(bad)])


# ---- per-input significance -------------------------------------------------
from repeatability_evals.stats import permutation_pvalue  # noqa: E402


def test_permutation_pvalue_basics():
    assert permutation_pvalue([1.0] * 6, [0.0] * 3) == pytest.approx(0.5 / 84)   # exact, mid-p
    assert permutation_pvalue([0.5] * 6, [0.5] * 3) == 0.5                        # all ties
    assert permutation_pvalue([1.0], [0.0]) is None                               # untestable
    assert permutation_pvalue([], [0.0, 1.0]) is None
    big = permutation_pvalue([1.0] * 20, [0.0] * 20, n_perm=500, seed=1)
    assert big is not None and big < 0.01


def noisy(rng, key, p, n):
    return [f"a{key}" if rng.random() > p else f"a{key}-{rng.randrange(3)}" for _ in range(n)]


def test_noisy_but_unchanged_system_has_few_inputs_flagged():
    import random
    rng = random.Random(0)
    base = {k: noisy(rng, k, 0.3, 4) for k in range(150)}
    cand = {k: noisy(rng, k, 0.3, 4) for k in range(150)}
    gated = compare_versions(base, cand, ExactMatch())
    ungated = compare_versions(base, cand, ExactMatch(), alpha=1.0)
    assert gated.share_changed < 0.08 and ungated.share_changed > 0.15
    assert gated.verdict == "no_meaningful_change"
    assert all(i.p_value is not None for i in gated.inputs)


def test_real_change_is_still_flagged_with_the_test():
    import random
    rng = random.Random(1)
    base = {k: noisy(rng, k, 0.1, 5) for k in range(60)}
    cand = {k: noisy(rng, f"{k}new" if k < 20 else k, 0.1, 5) for k in range(60)}
    r = compare_versions(base, cand, ExactMatch())
    flagged = {i.key for i in r.inputs if i.changed}
    assert r.verdict == "changed" and len(flagged & set(range(20))) >= 18
    assert len(flagged - set(range(20))) <= 2   # e.g. an input whose candidate became deterministic
