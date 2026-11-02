"""Simulated demo: compare two versions of a Q&A system with no gold labels.

No API key needed. The "system" and the "judge" are scripted stand-ins so the demo is
deterministic; in real use, `judge` is a call to your LLM and the runs come from your system.
Run:  python examples/compare_versions_demo.py
"""
import json
import random
import re

from repeatability_evals import DEFAULT_DIMENSIONS, RubricJudge, compare_versions

rng = random.Random(7)
POLICIES = ["gold", "silver", "bronze", "archive", "dr", "edge"]


def answer(policy, days, bullets, noisy):
    """One run of a pretend system. `noisy` runs sometimes reword (same meaning)."""
    lead = rng.choice(["Retention for", "The retention period of"]) if noisy else "Retention for"
    if noisy and rng.random() < 0.08:  # occasional flaky run: a wrong number
        days += 1
    body = f"{lead} {policy} is {days} days."
    return f"- {body}\n- Applies to all sites." if bullets else body


def version(days_for, bullets_for):
    runs = {}
    for q in range(120):
        p = POLICIES[q % len(POLICIES)]
        runs[f"q{q:03d}"] = [answer(p, days_for(q), bullets_for(q), True) for _ in range(4)]
    return runs


v1 = version(lambda q: 30, lambda q: False)
# v2 changes the retention number on ~12% of questions and switches ~20% to bulleted format.
v2 = version(lambda q: 90 if q % 8 == 0 else 30, lambda q: q % 5 == 0)


def judge(prompt):
    """Scripted stand-in for an LLM: scores each dimension by simple rules."""
    a = prompt.split("<response_a>")[1].split("</response_a>")[0]
    b = prompt.split("<response_b>")[1].split("</response_b>")[0]
    num = lambda t: re.findall(r"\d+", t)  # noqa: E731
    bullets = lambda t: t.lstrip().startswith("-")  # noqa: E731
    scores = {d.name: 5 for d in DEFAULT_DIMENSIONS}
    if num(a)[:1] != num(b)[:1]:
        scores["specifics"], scores["core_facts"] = 1, 3
    if bullets(a) != bullets(b):
        scores["format"], scores["supporting_detail"] = 1, 3
    why = "Retention period differs." if scores["specifics"] == 1 else (
        "Same facts; formatting differs." if scores["format"] == 1 else "Equivalent.")
    return json.dumps({"reason": why, "scores": scores})


report = compare_versions(v1, v2, RubricJudge(judge), names=("v1", "v2"))
print(report.to_markdown(top=2))
report.save("examples/demo_report.md")
