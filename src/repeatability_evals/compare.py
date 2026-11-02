"""Compare two versions of a system (or two configs, models, prompts) without gold labels.

For every input you give N runs from the *baseline* and M runs from the *candidate*. We measure

* how much the baseline agrees with itself (its run-to-run noise),
* how much the candidate agrees with itself,
* how much baseline and candidate agree with each other (the cross score),

and report **drift = expected - cross**, where *expected* is the cross score you would see if the
two versions were really the same system (the average of the two self scores). Subtracting the
noise floor is what separates "the system changed" from "the system is just non-deterministic".
No ground truth is needed, so it works on any data you can run, including a customer's own.
"""
from __future__ import annotations

import asyncio
import html as _html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .metrics import ON_ERROR, PairwiseConsistency, call_limit
from .sampling import CrossSampler, CycleSampler, PairSampler
from .similarity.base import as_similarity
from .stats import bootstrap_ci, mean, permutation_pvalue
from .types import ConsistencyResult

LOW_SIGNAL = 0.6  # if even the "no change" expectation is this low, the input is too noisy to judge


@dataclass
class InputComparison:
    key: Hashable
    n_baseline: int
    n_candidate: int
    baseline_self: Optional[float]  # None if fewer than 2 baseline runs
    candidate_self: Optional[float]
    cross: float
    expected: float  # cross score expected if nothing changed
    drift: float  # expected - cross; > 0 means the candidate differs more than noise explains
    changed: bool  # drift > tolerance and (when testable) p_value <= alpha
    p_value: Optional[float] = None  # permutation test: are cross pairs less alike than within pairs?
    flags: List[str] = field(default_factory=list)
    dimension_drift: Optional[Dict[str, float]] = None  # per-rubric-dimension drift
    driver: Optional[str] = None  # the dimension that moved most, if a rubric judge was used
    example: Optional[Dict[str, Any]] = None  # the least-similar baseline/candidate pair

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["key"] = str(self.key)
        if not include_text and d.get("example"):
            d["example"] = {k: v for k, v in d["example"].items() if k not in ("baseline", "candidate")}
        return d


@dataclass
class VersionComparison:
    n_inputs: int
    baseline_self: Optional[float]
    candidate_self: Optional[float]
    cross: float
    drift: float  # mean per-input drift
    drift_ci: Tuple[float, float]
    share_changed: float
    verdict: str  # "no_meaningful_change" | "changed" | "inconclusive"
    tolerance: float
    acceptable_drift: float
    alpha: float
    driver_counts: Dict[str, int]
    dimension_drift: Optional[Dict[str, float]]
    inputs: List[InputComparison]
    only_in_baseline: List[str] = field(default_factory=list)
    only_in_candidate: List[str] = field(default_factory=list)
    failed_inputs: Dict[str, str] = field(default_factory=dict)  # only with on_error="skip"
    baseline_name: str = "baseline"
    candidate_name: str = "candidate"

    # ---- querying -------------------------------------------------------------------------
    def most_changed(self, k: int = 10) -> List[InputComparison]:
        return sorted(self.inputs, key=lambda c: c.drift, reverse=True)[:k]

    def changed_inputs(self) -> List[InputComparison]:
        return [c for c in self.most_changed(len(self.inputs)) if c.changed]

    # ---- reports --------------------------------------------------------------------------
    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "inputs"}
        d["drift_ci"] = list(self.drift_ci)
        d["inputs"] = [c.to_dict(include_text) for c in self.inputs]
        return d

    def to_json(self, include_text: bool = True, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_text), indent=indent, default=str)

    def to_markdown(self, top: int = 10, include_text: bool = True, max_chars: int = 300) -> str:
        return _markdown(self, top, include_text, max_chars)

    def to_html(self, top: int = 10, include_text: bool = True, max_chars: int = 300) -> str:
        """A self-contained HTML page (no external assets) with the same content as the markdown."""
        body = _markdown_to_html(_markdown(self, top, include_text, max_chars))
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_html.escape(self.baseline_name)} vs {_html.escape(self.candidate_name)}</title>"
            f"<style>{_CSS}</style><body class='{self.verdict}'>{body}</body></html>"
        )

    def save(self, path: str, **kwargs: Any) -> None:
        """Write the report; format chosen by extension (.md, .html, .json)."""
        text = (
            self.to_json(**{k: v for k, v in kwargs.items() if k == "include_text"})
            if path.endswith(".json")
            else self.to_html(**kwargs) if path.endswith(".html")
            else self.to_markdown(**kwargs)
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


_VERDICT_TEXT = {
    "no_meaningful_change": "NO MEANINGFUL CHANGE: the candidate differs from the baseline no more "
    "than the baseline differs from itself.",
    "changed": "CHANGED: the candidate's outputs differ from the baseline's by more than "
    "run-to-run noise explains.",
    "inconclusive": "INCONCLUSIVE: mixed evidence (some inputs changed, or too little data). "
    "Review the changed inputs below, or add inputs / runs.",
}


_CSS = (
    ":root{color-scheme:light dark;--fg:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--bg:#fff;"
    "--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e}"
    "@media (prefers-color-scheme:dark){:root{--fg:#f5f5f7;--muted:#a1a1a6;--line:#3a3a3c;"
    "--bg:#1c1c1e;--ok:#3fb950;--warn:#d29922;--bad:#f85149}}"
    "body{font:15px/1.55 system-ui,-apple-system,sans-serif;color:var(--fg);background:var(--bg);"
    "max-width:920px;margin:2rem auto;padding:0 16px}"
    "h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid var(--line)}"
    "h3{font-size:1rem;margin-top:1.4rem}table{border-collapse:collapse;margin:.6rem 0;"
    "display:block;overflow-x:auto}td,th{border:1px solid var(--line);padding:4px 10px;text-align:left}"
    "code{font-size:.9em}em{color:var(--muted)}"
    "body.no_meaningful_change p.verdict{color:var(--ok)}body.changed p.verdict{color:var(--bad)}"
    "body.inconclusive p.verdict{color:var(--warn)}"
)

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"(?<![\w*])[_*]([^_*]+)[_*](?![\w*])"), r"<em>\1</em>"),
]


def _inline(text: str) -> str:
    out = _html.escape(text, quote=False)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _markdown_to_html(md: str) -> str:
    """Render the small markdown subset used by the report (headings, tables, bullets)."""
    html: List[str] = []
    table: List[List[str]] = []
    bullets: List[str] = []

    def flush() -> None:
        if table:
            head, *rows = table
            html.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head)
                        + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                                  for r in rows)
                        + "</tbody></table>")
            table.clear()
        if bullets:
            html.append("<ul>" + "".join(f"<li>{_inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(re.fullmatch(r"-+", c) for c in cells if c):
                table.append(cells)
            continue
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
            continue
        flush()
        if not stripped:
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= level <= 3 and stripped[level:level + 1] == " ":
            html.append(f"<h{level}>{_inline(stripped[level + 1:])}</h{level}>")
        elif stripped.startswith("**"):
            html.append(f"<p class='verdict'>{_inline(stripped)}</p>")
        else:
            html.append(f"<p>{_inline(stripped)}</p>")
    flush()
    return "\n".join(html)


def _f(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _clip(text: Any, n: int) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _markdown(c: VersionComparison, top: int, include_text: bool, max_chars: int) -> str:
    lo, hi = c.drift_ci
    out = [
        f"# {c.baseline_name} vs {c.candidate_name}: consistency report",
        "",
        f"**{_VERDICT_TEXT[c.verdict]}**",
        "",
        f"{c.n_inputs} inputs compared; {c.share_changed:.0%} changed beyond the noise floor "
        f"(per-input drift > {c.tolerance:g} and beyond run-to-run noise, p <= {c.alpha:g}).",
        "",
        "| | score (1 = identical) |",
        "|---|---|",
        f"| {c.baseline_name} vs itself (noise floor) | {_f(c.baseline_self)} |",
        f"| {c.candidate_name} vs itself | {_f(c.candidate_self)} |",
        f"| {c.baseline_name} vs {c.candidate_name} | {_f(c.cross)} |",
        f"| **drift** (expected - cross; 0 = no change) | **{c.drift:+.3f}** "
        f"(95% CI {lo:+.3f} to {hi:+.3f}) |",
        "",
    ]
    if c.dimension_drift:
        out += ["## What changed", "", "| dimension | drift |", "|---|---|"]
        out += [f"| {k} | {v:+.3f} |" for k, v in sorted(c.dimension_drift.items(), key=lambda kv: -kv[1])]
        if c.driver_counts:
            parts = ", ".join(f"{n} mainly in `{k}`" for k, n in
                              sorted(c.driver_counts.items(), key=lambda kv: -kv[1]))
            out += ["", f"Changed inputs by main driver: {parts}."]
        out.append("")
    changed = c.changed_inputs()
    out += [f"## Most changed inputs (showing {min(top, len(changed))} of {len(changed)} changed)", ""]
    if not changed:
        out += ["None: no input exceeded the drift tolerance.", ""]
    for item in changed[:top]:
        p = "" if item.p_value is None else f", p = {item.p_value:.3f}"
        out += [f"### `{item.key}`: drift {item.drift:+.2f} "
                f"(cross {item.cross:.2f} vs expected {item.expected:.2f}{p})", ""]
        if item.driver:
            out.append(f"- Main driver: `{item.driver}`")
        if item.flags:
            out.append(f"- Flags: {', '.join(item.flags)}")
        ex = item.example or {}
        if ex.get("reason"):
            out.append(f"- Judge: {_clip(ex['reason'], max_chars)}")
        if include_text and "baseline" in ex:
            out += [f"- {c.baseline_name}: {_clip(ex['baseline'], max_chars)}",
                    f"- {c.candidate_name}: {_clip(ex['candidate'], max_chars)}"]
        out.append("")
    flagged = [i for i in c.inputs if "noise_floor_unknown" in i.flags]
    if flagged:
        out += [f"_{len(flagged)} inputs have fewer than 2 runs on a side, so their noise floor is "
                "unknown and any difference counts as drift. Use 3+ runs per side for a fair test._", ""]
    if c.only_in_baseline or c.only_in_candidate:
        out += [f"_Skipped: {len(c.only_in_baseline)} inputs only in {c.baseline_name}, "
                f"{len(c.only_in_candidate)} only in {c.candidate_name}._", ""]
    out += [
        "## How to read this", "",
        "No ground truth is used. Consistency is not correctness: a version can agree with the baseline "
        "and both be wrong, or differ and be better. Drift tells you *where to look*; judge whether the "
        "changes are improvements. Drift subtracts the baseline's own run-to-run noise, so a "
        "non-deterministic system is not flagged against itself.",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
def _score_or_none(r: ConsistencyResult) -> Optional[float]:
    return None if r.details.get("degenerate") else r.score


def _dims(r: ConsistencyResult) -> Optional[Dict[str, float]]:
    return r.details.get("dimensions")


def _assemble(
    key: Hashable, a: List[Any], b: List[Any], ra: ConsistencyResult, rb: ConsistencyResult,
    rx: ConsistencyResult, tolerance: float, with_examples: bool, alpha: float = 0.2, seed: int = 0,
) -> InputComparison:
    sa, sb = _score_or_none(ra), _score_or_none(rb)
    known = [s for s in (sa, sb) if s is not None]
    flags: List[str] = []
    if len(known) < 2:
        flags.append("noise_floor_unknown")
    expected = mean(known) if known else 1.0
    drift = expected - rx.score
    if expected < LOW_SIGNAL:
        flags.append("low_signal")  # both versions are so noisy that small drifts are unreadable
    dim_drift = driver = None
    dx = _dims(rx)
    if dx:
        refs = [d for d in (_dims(ra), _dims(rb)) if d]
        if refs:
            dim_drift = {k: mean([d[k] for d in refs]) - dx[k] for k in dx}
        else:
            dim_drift = {k: 1.0 - v for k, v in dx.items()}
        driver = max(dim_drift, key=lambda k: dim_drift[k])
    example = None
    if rx.pairs:
        w = min(rx.pairs, key=lambda p: p.score)
        example = {"score": w.score, "reason": w.reason}
        if with_examples:
            example["baseline"], example["candidate"] = a[w.i], b[w.j - len(a)]
    p_value = permutation_pvalue([p.score for p in ra.pairs + rb.pairs], [p.score for p in rx.pairs],
                                 seed=seed)
    changed = drift > tolerance and (p_value is None or p_value <= alpha)
    return InputComparison(
        key, len(a), len(b), sa, sb, rx.score, expected, drift, changed, p_value, flags,
        dim_drift, driver, example,
    )


def _summarise(
    items: List[InputComparison], tolerance: float, acceptable_drift: float, max_changed_share: float,
    alpha: float, seed: int, only_a: List[str], only_b: List[str], names: Tuple[str, str],
    failed: Optional[Dict[str, str]] = None,
) -> VersionComparison:
    if not items:
        if failed:
            raise ValueError(f"every input failed ({len(failed)}); first: {next(iter(failed.values()))}")
        raise ValueError("no inputs are present in both baseline and candidate")
    drifts = [i.drift for i in items]
    lo, hi = bootstrap_ci(drifts, seed=seed)
    share = sum(i.changed for i in items) / len(items)
    if lo > acceptable_drift:
        verdict = "changed"
    elif hi <= acceptable_drift and share <= max_changed_share:
        verdict = "no_meaningful_change"
    else:
        verdict = "inconclusive"
    sa = [i.baseline_self for i in items if i.baseline_self is not None]
    sb = [i.candidate_self for i in items if i.candidate_self is not None]
    dims = [i.dimension_drift for i in items if i.dimension_drift]
    counts: Dict[str, int] = {}
    for i in items:
        if i.changed and i.driver:
            counts[i.driver] = counts.get(i.driver, 0) + 1
    return VersionComparison(
        n_inputs=len(items),
        baseline_self=mean(sa) if sa else None,
        candidate_self=mean(sb) if sb else None,
        cross=mean([i.cross for i in items]),
        drift=mean(drifts), drift_ci=(lo, hi), share_changed=share, verdict=verdict,
        tolerance=tolerance, acceptable_drift=acceptable_drift, alpha=alpha, driver_counts=counts,
        dimension_drift={k: mean([d[k] for d in dims]) for k in dims[0]} if dims else None,
        inputs=items, only_in_baseline=only_a, only_in_candidate=only_b,
        failed_inputs=dict(failed or {}), baseline_name=names[0], candidate_name=names[1],
    )


def _setup(
    baseline: Mapping[Hashable, Sequence[Any]], candidate: Mapping[Hashable, Sequence[Any]],
    similarity: Any, preprocess: Any, within_sampler: Optional[PairSampler],
    cross_budget: Optional[int], concurrency: int, seed: int, on_error: str = "raise",
):
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}")
    sim = as_similarity(similarity)
    within = PairwiseConsistency(sim, sampler=within_sampler or CycleSampler(seed=seed),
                                 preprocess=preprocess, concurrency=concurrency, on_error=on_error)
    keys = [k for k in baseline if k in candidate]
    only_a = [str(k) for k in baseline if k not in candidate]
    only_b = [str(k) for k in candidate if k not in baseline]

    def cross_metric(n_a: int) -> PairwiseConsistency:
        return PairwiseConsistency(sim, sampler=CrossSampler(n_a, cross_budget, seed),
                                   preprocess=preprocess, concurrency=concurrency, on_error=on_error)
    return within, cross_metric, keys, only_a, only_b


def compare_versions(
    baseline: Mapping[Hashable, Sequence[Any]],
    candidate: Mapping[Hashable, Sequence[Any]],
    similarity: Any,
    *,
    contexts: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    preprocess: Any = None,
    within_sampler: Optional[PairSampler] = None,
    cross_budget: Optional[int] = None,
    tolerance: float = 0.1,
    alpha: float = 0.2,
    acceptable_drift: float = 0.05,
    max_changed_share: float = 0.15,
    include_examples: bool = True,
    names: Tuple[str, str] = ("baseline", "candidate"),
    seed: int = 0,
    on_error: str = "raise",
) -> VersionComparison:
    """Compare two versions on the same inputs, no gold labels needed.

    ``baseline`` / ``candidate`` map an input key to that version's runs (outputs, retrieved IDs,
    traces...) for that input; give 3+ runs per side so the noise floor can be estimated.
    ``similarity`` is any ``Similarity`` or ``fn(a, b) -> [0, 1]``; ``contexts`` optionally maps
    keys to ``{"query": ...}`` for an LLM judge.

    ``tolerance``: an input counts as changed when its drift exceeds this **and**, when there are
    enough compared pairs to test it, a permutation test finds its cross-version pairs less alike
    than its within-version pairs (``p_value <= alpha``). The test is what keeps a noisy system
    from having many inputs flagged by chance; it needs about 3+ runs per side to apply.
    ``acceptable_drift``: the mean drift you are willing to ship; the verdict is
    ``no_meaningful_change`` if the 95% CI of mean drift stays below it (and at most
    ``max_changed_share`` of inputs changed), ``changed`` if the CI is entirely above it,
    otherwise ``inconclusive``.

    ``on_error="skip"``: a failing comparison (e.g. an unparsable judge reply or an API
    timeout) is dropped, and an input whose comparisons all fail is listed in
    ``failed_inputs`` instead of aborting the run. Default ``"raise"``.
    """
    within, cross_metric, keys, only_a, only_b = _setup(
        baseline, candidate, similarity, preprocess, within_sampler, cross_budget, 8, seed, on_error)
    items: List[InputComparison] = []
    failed: Dict[str, str] = {}
    for k in keys:
        a, b = list(baseline[k]), list(candidate[k])
        if not a or not b:
            raise ValueError(f"input {k!r} has no runs on one side")
        ctx = (contexts or {}).get(k)
        try:
            ra, rb = within.evaluate(a, context=ctx), within.evaluate(b, context=ctx)
            rx = cross_metric(len(a)).evaluate(a + b, context=ctx)
        except Exception as e:
            if on_error == "raise":
                raise
            failed[str(k)] = repr(e)
            continue
        items.append(_assemble(k, a, b, ra, rb, rx, tolerance, include_examples, alpha, seed))
    return _summarise(items, tolerance, acceptable_drift, max_changed_share, alpha, seed, only_a, only_b,
                      names, failed)


async def acompare_versions(
    baseline: Mapping[Hashable, Sequence[Any]],
    candidate: Mapping[Hashable, Sequence[Any]],
    similarity: Any,
    *,
    contexts: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
    preprocess: Any = None,
    within_sampler: Optional[PairSampler] = None,
    cross_budget: Optional[int] = None,
    tolerance: float = 0.1,
    alpha: float = 0.2,
    acceptable_drift: float = 0.05,
    max_changed_share: float = 0.15,
    include_examples: bool = True,
    names: Tuple[str, str] = ("baseline", "candidate"),
    seed: int = 0,
    concurrency: int = 8,
    on_error: str = "raise",
) -> VersionComparison:
    """Async ``compare_versions``. ``concurrency`` caps simultaneous similarity (LLM) calls
    across the whole comparison, so it can be set to your provider's rate limit."""
    within, cross_metric, keys, only_a, only_b = _setup(
        baseline, candidate, similarity, preprocess, within_sampler, cross_budget, concurrency, seed,
        on_error)
    for k in keys:
        if not baseline[k] or not candidate[k]:
            raise ValueError(f"input {k!r} has no runs on one side")
    inputs_in_flight = asyncio.Semaphore(max(1, concurrency))

    async def one(k: Hashable) -> Any:
        a, b = list(baseline[k]), list(candidate[k])
        ctx = (contexts or {}).get(k)
        async with inputs_in_flight:
            try:
                ra, rb, rx = await asyncio.gather(
                    within.aevaluate(a, context=ctx), within.aevaluate(b, context=ctx),
                    cross_metric(len(a)).aevaluate(a + b, context=ctx))
            except Exception as e:
                if on_error == "raise":
                    raise
                return str(k), repr(e)
        return _assemble(k, a, b, ra, rb, rx, tolerance, include_examples, alpha, seed)

    with call_limit(concurrency):
        out = await asyncio.gather(*(one(k) for k in keys))
    items = [x for x in out if isinstance(x, InputComparison)]
    failed = dict(x for x in out if not isinstance(x, InputComparison))
    return _summarise(items, tolerance, acceptable_drift, max_changed_share, alpha, seed, only_a, only_b,
                      names, failed)
