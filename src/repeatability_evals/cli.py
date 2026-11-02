"""Command line: compare two versions of a system from saved outputs, no gold labels.

    repeatability-evals compare baseline.json candidate.json --out report.md

Input files are JSON: ``{"<input id>": ["run 1 output", "run 2 output", ...]}`` or
``{"<input id>": {"query": "...", "outputs": [...]}}``. By default outputs are compared with
n-gram overlap (no model needed, runs fully offline). For semantic comparison pass
``--judge package.module:function`` where the function takes a prompt string and returns the
model's reply (it may be ``async``); add ``--rubric`` for the multi-dimension judge.
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


def load_runs(path: str) -> Tuple[Dict[str, List[Any]], Dict[str, Dict[str, Any]]]:
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
        else:
            runs[key] = list(val)
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
    c.add_argument("--acceptable-drift", type=float, default=0.05, help="mean drift you would ship")
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--no-text", action="store_true", help="omit raw outputs from the report")
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
    kwargs: Dict[str, Any] = dict(contexts=contexts, tolerance=args.tolerance,
                                  acceptable_drift=args.acceptable_drift, names=tuple(args.names),
                                  include_examples=not args.no_text)
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
    return {"no_meaningful_change": 0, "inconclusive": 1, "changed": 2}[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
