"""Command line: compare two versions of a system from saved outputs, no gold labels.

    repeatability-evals compare baseline.json candidate.json --out report.md

Input files are JSON: ``{"<input id>": ["run 1 output", "run 2 output", ...]}`` or
``{"<input id>": {"query": "...", "outputs": [...]}}``, or JSON Lines (``.jsonl``) with one run
per line: ``{"id": "...", "output": "...", "query": "..."}``, the shape most request logs already
have. By default outputs are compared with
n-gram overlap (no model needed, runs fully offline). For semantic comparison pass
``--judge package.module:function`` where the function takes a prompt string and returns the
model's reply (it may be ``async``); add ``--rubric`` for the multi-dimension judge.

Exit status: 0 no meaningful change, 3 inconclusive, 4 changed (1 and 2 are left to Python and
argparse for errors), so the command can gate a CI pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from .compare import acompare_versions, compare_versions
from .similarity import EditSimilarity, ExactMatch, LLMJudge, NgramJaccard, RougeL, RubricJudge

EXIT_CODES = {"no_meaningful_change": 0, "inconclusive": 3, "changed": 4}


def _load_jsonl(path: str) -> Tuple[Dict[str, List[Any]], Dict[str, Dict[str, Any]]]:
    runs: Dict[str, List[Any]] = {}
    contexts: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{path}:{n}: invalid JSON ({e})") from e
            if not isinstance(row, dict) or "id" not in row or not ("output" in row or "outputs" in row):
                raise SystemExit(f'{path}:{n}: each line needs "id" and "output" (or "outputs")')
            key = str(row["id"])
            outs = row["outputs"] if "outputs" in row else [row["output"]]
            runs.setdefault(key, []).extend(outs)
            if row.get("query") is not None:
                contexts.setdefault(key, {"query": row["query"]})
    return runs, contexts


def load_runs(path: str) -> Tuple[Dict[str, List[Any]], Dict[str, Dict[str, Any]]]:
    """Read runs from ``.json`` (object keyed by input id) or ``.jsonl`` (one run per line)."""
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object mapping input id to runs")
    runs: Dict[str, List[Any]] = {}
    contexts: Dict[str, Dict[str, Any]] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            runs[key] = list(val.get("outputs", []))
            if val.get("query") is not None:
                contexts[key] = {"query": val["query"]}
        elif isinstance(val, list):
            runs[key] = list(val)
        else:
            raise SystemExit(f"{path}: input {key!r} must map to a list of runs or an object")
    return runs, contexts


def _import_callable(spec: str) -> Callable[..., Any]:
    mod, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--judge must look like package.module:function")
    sys.path.insert(0, ".")
    return getattr(importlib.import_module(mod), attr)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="repeatability-evals", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="compare a candidate version against a baseline")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--similarity", choices=["ngram", "rouge", "edit", "exact"], default="ngram",
                   help="offline similarity used when no --judge is given (default: ngram)")
    c.add_argument("--judge", metavar="MODULE:FUNC", help="LLM completion function (prompt -> reply)")
    c.add_argument("--rubric", action="store_true", help="use the 5-dimension RubricJudge")
    c.add_argument("--symmetric", action="store_true", help="judge both orders (2x cost)")
    c.add_argument("--tolerance", type=float, default=0.1, help="per-input drift counted as changed")
    c.add_argument("--alpha", type=float, default=0.2,
                   help="per-input significance level for 'changed' (1.0 = drift threshold only)")
    c.add_argument("--acceptable-drift", type=float, default=0.05, help="mean drift you would ship")
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--no-text", action="store_true", help="omit raw outputs from the report")
    c.add_argument("--on-error", choices=["raise", "skip"], default="raise",
                   help="skip comparisons that fail (bad judge reply, timeout) instead of stopping")
    c.add_argument("--names", nargs=2, metavar=("BASE", "CAND"), default=["baseline", "candidate"])
    c.add_argument("--out", help="write report here (.md, .html or .json); default: print markdown")
    args = ap.parse_args(argv)

    base, ctx_a = load_runs(args.baseline)
    cand, ctx_b = load_runs(args.candidate)
    contexts = {**ctx_b, **ctx_a}
    judge_fn = _import_callable(args.judge) if args.judge else None
    if judge_fn is not None:
        cls = RubricJudge if args.rubric else LLMJudge
        kw: Dict[str, Any] = {"symmetric": args.symmetric}
        sim = (cls(allm=judge_fn, **kw) if inspect.iscoroutinefunction(judge_fn)
               else cls(judge_fn, **kw))
    else:
        sim = {"ngram": NgramJaccard(2), "rouge": RougeL(), "edit": EditSimilarity(),
               "exact": ExactMatch()}[args.similarity]
    kwargs: Dict[str, Any] = dict(contexts=contexts, tolerance=args.tolerance, alpha=args.alpha,
                                  acceptable_drift=args.acceptable_drift, names=tuple(args.names),
                                  include_examples=not args.no_text, on_error=args.on_error)
    if judge_fn is not None:
        report = asyncio.run(acompare_versions(base, cand, sim, concurrency=args.concurrency, **kwargs))
    else:
        report = compare_versions(base, cand, sim, **kwargs)
    if args.out:
        report.save(args.out, include_text=not args.no_text)
        print(f"{report.verdict}: drift {report.drift:+.3f}, "
              f"{report.share_changed:.0%} of {report.n_inputs} inputs changed -> {args.out}")
    else:
        print(report.to_markdown(include_text=not args.no_text))
    if report.failed_inputs:
        print(f"warning: {len(report.failed_inputs)} inputs failed and were skipped", file=sys.stderr)
    return EXIT_CODES[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
