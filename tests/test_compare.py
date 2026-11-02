import asyncio
import json
import random

import pytest

from repeatability_evals import (
    DEFAULT_DIMENSIONS,
    CrossSampler,
    ExactMatch,
    NgramJaccard,
    RubricJudge,
    acompare_versions,
    compare_versions,
)


# ---- CrossSampler ---------------------------------------------------------
@pytest.mark.parametrize("n_a,n_b", [(3, 3), (5, 3), (2, 6), (1, 4)])
def test_cross_sampler_only_cross_pairs_and_covers_every_run(n_a, n_b):
    pairs = CrossSampler(n_a, seed=1).sample(n_a + n_b)
    assert len(pairs) == len(set(pairs)) == min(max(n_a, n_b), n_a * n_b)
    assert all(i < n_a <= j for i, j in pairs)
    assert {i for i, _ in pairs} == set(range(n_a))
    assert {j for _, j in pairs} == set(range(n_a, n_a + n_b))


def test_cross_sampler_budgets():
    assert len(CrossSampler(4, budget=99).sample(8)) == 16          # all pairs
    assert len(CrossSampler(4, budget=6, seed=2).sample(8)) == 6    # topped up, distinct
    assert len(set(CrossSampler(4, budget=6, seed=2).sample(8))) == 6
    assert len(CrossSampler(4, budget=2).sample(8)) == 2
    assert CrossSampler(4).sample(4) == []
    assert CrossSampler(4, seed=3).sample(8) == CrossSampler(4, seed=3).sample(8)


# ---- compare_versions -----------------------------------------------------
def noisy_system(rng, truth, p_flip):
    """Answers 'ans-<truth>' but flips to a random variant with probability p_flip per run."""
    return [f"ans-{truth}" if rng.random() > p_flip else f"ans-{truth}-v{rng.randrange(50)}"
            for _ in range(5)]


def make(n_inputs, p_base, p_cand, changed_share=0.0, seed=0):
    rng = random.Random(seed)
    base, cand = {}, {}
    for k in range(n_inputs):
        base[k] = noisy_system(rng, k, p_base)
        shifted = rng.random() < changed_share
        cand[k] = noisy_system(rng, f"{k}-new" if shifted else k, p_cand)
    return base, cand


def test_identical_noisy_systems_show_no_drift():
    base, cand = make(60, 0.3, 0.3, seed=1)
    r = compare_versions(base, cand, ExactMatch())
    assert r.verdict == "no_meaningful_change"
    assert abs(r.drift) < 0.1 and r.baseline_self < 0.95        # noise is real, but not flagged


def test_noise_floor_prevents_false_alarm_vs_naive_cross_score():
    base, cand = make(60, 0.5, 0.5, seed=2)
    r = compare_versions(base, cand, ExactMatch())
    assert r.cross < 0.8                      # naive "agreement with baseline" looks bad...
    assert r.verdict != "changed"             # ...but it's just the system's own noise


def test_real_change_is_detected_and_localised():
    base, cand = make(60, 0.1, 0.1, changed_share=0.5, seed=3)
    r = compare_versions(base, cand, ExactMatch())
    assert r.verdict == "changed"
    assert 0.3 < r.share_changed < 0.7
    shifted = {c.key for c in r.changed_inputs()}
    assert all(str(c.example["baseline"]).startswith("ans-") for c in r.changed_inputs())
    assert r.most_changed(3)[0].key in shifted


def test_deterministic_candidate_vs_noisy_baseline():
    base, _ = make(30, 0.4, 0.0, seed=4)
    cand = {k: [f"ans-{k}"] * 3 for k in base}
    r = compare_versions(base, cand, ExactMatch())
    assert r.candidate_self == 1.0 and r.baseline_self < 1.0


def test_single_run_per_side_flags_unknown_noise_floor():
    r = compare_versions({"q": ["a"]}, {"q": ["b"]}, ExactMatch())
    assert "noise_floor_unknown" in r.inputs[0].flags
    assert r.inputs[0].drift == 1.0 and r.inputs[0].changed


def test_keys_missing_on_one_side_are_reported_and_empty_overlap_errors():
    r = compare_versions({"a": ["x", "x"], "b": ["y", "y"]}, {"a": ["x", "x"], "c": ["z"]}, ExactMatch())
    assert r.n_inputs == 1 and r.only_in_baseline == ["b"] and r.only_in_candidate == ["c"]
    with pytest.raises(ValueError):
        compare_versions({"a": ["x"]}, {"b": ["x"]}, ExactMatch())
    with pytest.raises(ValueError):
        compare_versions({"a": []}, {"a": ["x"]}, ExactMatch())


def test_preprocess_and_text_similarity():
    base = {"q": [{"t": "alpha beta gamma delta"}] * 3}
    cand = {"q": [{"t": "alpha beta gamma delta"}, {"t": "alpha beta gamma delta"}]}
    r = compare_versions(base, cand, NgramJaccard(2), preprocess=lambda x: x["t"])
    assert r.drift == 0.0 and not r.inputs[0].changed


def test_rubric_judge_gives_per_dimension_drift_and_driver():
    def llm(prompt):
        scores = {d.name: 5 for d in DEFAULT_DIMENSIONS}
        if "NEWSTYLE" in prompt and "OLDSTYLE" in prompt:        # cross pair: only format differs
            scores["format"] = 1
        return json.dumps({"reason": "formatting differs", "scores": scores})
    base = {k: ["OLDSTYLE answer"] * 3 for k in range(6)}
    cand = {k: ["NEWSTYLE answer"] * 3 for k in range(6)}
    r = compare_versions(base, cand, RubricJudge(llm), tolerance=0.05)
    assert r.dimension_drift["format"] == pytest.approx(1.0)
    assert r.dimension_drift["core_facts"] == 0.0
    assert r.driver_counts == {"format": 6}
    assert "mainly in `format`" in r.to_markdown()
    assert r.changed_inputs()[0].example["reason"] == "formatting differs"


def test_reports_render_and_can_redact_customer_text(tmp_path):
    base = {"q1": ["SECRET-OLD"] * 3, "q2": ["same"] * 3}
    cand = {"q1": ["SECRET-NEW"] * 3, "q2": ["same"] * 3}
    r = compare_versions(base, cand, ExactMatch(), names=("v1", "v2"))
    md = r.to_markdown()
    assert "v1 vs v2" in md and "SECRET-OLD" in md and "`q1`" in md and "How to read this" in md
    safe = r.to_markdown(include_text=False)
    assert "SECRET" not in safe and "`q1`" in safe
    assert "SECRET" not in r.to_json(include_text=False)
    assert json.loads(r.to_json())["n_inputs"] == 2
    for ext in ("md", "html", "json"):
        r.save(str(tmp_path / f"r.{ext}"))
        assert (tmp_path / f"r.{ext}").read_text().strip()


def test_html_escapes_content():
    r = compare_versions({"q": ["<script>x</script>"] * 3}, {"q": ["other"] * 3}, ExactMatch())
    assert "<script>" not in r.to_html()


def test_async_matches_sync():
    base, cand = make(15, 0.2, 0.2, changed_share=0.3, seed=5)
    s = compare_versions(base, cand, ExactMatch())
    a = asyncio.run(acompare_versions(base, cand, ExactMatch(), concurrency=4))
    assert s.drift == pytest.approx(a.drift) and s.verdict == a.verdict
    assert [c.key for c in s.most_changed(5)] == [c.key for c in a.most_changed(5)]
