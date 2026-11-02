import asyncio

import pytest

from consistency_evals import JudgeParseError, LLMJudge, PairwiseConsistency
from consistency_evals.similarity.judge import parse_judge_output

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
    from consistency_evals import AllPairs
    metric.sampler = AllPairs()
    r = metric.evaluate(["same", "same", "outlier"], context={"query": "q"})
    assert r.n_pairs == 3
    assert r.score == pytest.approx((1 + 0 + 0) / 3)
    assert r.details["agreement_rate"] == pytest.approx(1 / 3)
    assert any(p.reason == "differs" for p in r.pairs)
