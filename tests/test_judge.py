import asyncio
import json

import pytest

from repeatability_evals import JudgeParseError, LLMJudge, PairwiseConsistency
from repeatability_evals.similarity.judge import parse_judge_output

SCALE = (1.0, 5.0)


@pytest.mark.parametrize("reply,expected", [
    ('{"reason": "same facts", "score": 5}', (5.0, "same facts")),
    ('Sure! ```json\n{"score": 3, "reason": "partly"}\n```', (3.0, "partly")),
    ("Score: 4", (4.0, None)),
    ("I would give this a rating of 2 overall.", (2.0, None)),
    ("The answer is 4", (4.0, None)),
])
def test_parse_judge_output(reply, expected):
    assert parse_judge_output(reply, SCALE) == expected


def test_parse_failure():
    with pytest.raises(JudgeParseError):
        parse_judge_output("no idea, sorry", SCALE)


def test_judge_normalises_and_keeps_raw_and_reason():
    judge = LLMJudge(lambda p: '{"reason": "close", "score": 4}')
    r = judge.score("a", "b")
    assert (r.score, r.raw, r.reason) == (0.75, 4.0, "close")


def test_judge_clamps_out_of_scale_replies():
    assert LLMJudge(lambda p: '{"score": 9}')("a", "b") == 1.0
    assert LLMJudge(lambda p: '{"score": 0}')("a", "b") == 0.0


def test_prompt_contains_query_and_both_items_and_survives_json_braces():
    seen = []
    LLMJudge(lambda p: seen.append(p) or '{"score": 5}').score(
        "ALPHA", "BETA", context={"query": "What is X?"})
    prompt = seen[0]
    assert "What is X?" in prompt and "ALPHA" in prompt and "BETA" in prompt
    assert '{"reason"' in prompt            # literal JSON braces in the template are untouched
    seen.clear()
    LLMJudge(lambda p: seen.append(p) or '{"score": 5}').score("a", "b")
    assert "(not provided)" in seen[0]


def test_symmetric_averages_both_orders():
    def position_biased(prompt):             # always prefers whichever response is "A"
        return '{"score": 5}' if prompt.index("ALPHA") < prompt.index("BETA") else '{"score": 1}'
    one_way = LLMJudge(position_biased)
    both = LLMJudge(position_biased, symmetric=True)
    assert one_way("ALPHA", "BETA") == 1.0 and both("ALPHA", "BETA") == 0.5
    assert both("ALPHA", "BETA") == both("BETA", "ALPHA")


def test_custom_scale():
    judge = LLMJudge(lambda p: '{"score": 5}', scale=(0, 10))
    assert judge("a", "b") == 0.5


def test_constructor_validation():
    with pytest.raises(ValueError):
        LLMJudge()
    with pytest.raises(ValueError):
        LLMJudge(lambda p: "", scale=(5, 1))


def test_async_only_judge():
    async def allm(prompt):
        return '{"score": 5}'
    judge = LLMJudge(allm=allm)
    assert asyncio.run(judge.ascore("a", "b")).score == 1.0
    with pytest.raises(RuntimeError):
        judge.score("a", "b")


def test_sync_llm_is_usable_from_async_path():
    assert asyncio.run(LLMJudge(lambda p: '{"score": 3}').ascore("a", "b")).score == 0.5


def test_judge_inside_pairwise_metric_end_to_end():
    def llm(prompt):
        return '{"score": 5}' if "outlier" not in prompt else '{"score": 1, "reason": "differs"}'
    metric = PairwiseConsistency(LLMJudge(llm), agreement_threshold=0.75)
    from repeatability_evals import AllPairs
    metric.sampler = AllPairs()
    r = metric.evaluate(["same", "same", "outlier"], context={"query": "q"})
    assert r.n_pairs == 3
    assert r.score == pytest.approx((1 + 0 + 0) / 3)
    assert r.details["agreement_rate"] == pytest.approx(1 / 3)
    assert any(p.reason == "differs" for p in r.pairs)


# ---- RubricJudge ----------------------------------------------------------

from repeatability_evals import (  # noqa: E402
    DEFAULT_DIMENSIONS,
    AllPairs,
    Dimension,
    RubricJudge,
    build_rubric_prompt,
)
from repeatability_evals.similarity.judge import parse_rubric_output  # noqa: E402

NAMES = [d.name for d in DEFAULT_DIMENSIONS]


def reply(**over):
    scores = {n: 5 for n in NAMES}
    scores.update(over)
    return json.dumps({"reason": "r", "scores": scores})


def test_default_rubric_has_at_most_five_dimensions_and_prompt_lists_them():
    assert len(DEFAULT_DIMENSIONS) <= 5
    prompt = build_rubric_prompt()
    assert all(f"- {n}:" in prompt for n in NAMES) and '"scores"' in prompt
    assert "{item_a}" in prompt and "{query}" in prompt


def test_rubric_weighted_score_and_per_dimension_metadata():
    r = RubricJudge(lambda p: reply(format=1)).score("a", "b")
    assert r.metadata["dimensions"]["format"] == 0.0
    assert r.metadata["dimensions"]["core_facts"] == 1.0
    assert r.score == pytest.approx(0.90)            # format weight is 0.10
    assert r.raw == pytest.approx(1 + 0.90 * 4)


def test_rubric_core_difference_hurts_more_than_format_difference():
    core = RubricJudge(lambda p: reply(core_facts=1))("a", "b")
    fmt = RubricJudge(lambda p: reply(format=1))("a", "b")
    assert core < fmt


def test_rubric_parse_errors_and_clamping():
    with pytest.raises(JudgeParseError):
        parse_rubric_output("nope", NAMES, SCALE)
    with pytest.raises(JudgeParseError):
        parse_rubric_output('{"scores": {"core_facts": 5}}', NAMES, SCALE)
    out, _ = parse_rubric_output(reply(core_facts=9, format=0), NAMES, SCALE)
    assert out["core_facts"] == 5.0 and out["format"] == 1.0


def test_custom_dimensions_and_validation():
    dims = [Dimension("a", "first", 3), Dimension("b", "second", 1)]
    judge = RubricJudge(lambda p: '{"scores": {"a": 5, "b": 1}}', dimensions=dims)
    assert judge("x", "y") == pytest.approx(0.75)
    with pytest.raises(ValueError):
        RubricJudge(lambda p: "", dimensions=[])
    with pytest.raises(ValueError):
        RubricJudge(lambda p: "", dimensions=[Dimension("a", "d"), Dimension("a", "e")])
    with pytest.raises(ValueError):
        RubricJudge()


def test_rubric_symmetric_and_async():
    def biased(prompt):
        return reply(core_facts=5) if prompt.index("ALPHA") < prompt.index("BETA") else reply(core_facts=1)
    assert RubricJudge(biased, symmetric=True)("ALPHA", "BETA") == pytest.approx(1 - 0.4 * 0.5)

    async def allm(prompt):
        return reply()
    assert asyncio.run(RubricJudge(allm=allm).ascore("a", "b")).score == 1.0


def test_pairwise_reports_mean_per_dimension():
    def llm(prompt):
        return reply(format=1) if "outlier" in prompt else reply()
    m = PairwiseConsistency(RubricJudge(llm), sampler=AllPairs())
    r = m.evaluate(["same", "same", "outlier"])
    assert r.details["dimensions"]["core_facts"] == 1.0
    assert r.details["dimensions"]["format"] == pytest.approx(1 / 3)
    assert r.pairs[0].dimensions is not None
