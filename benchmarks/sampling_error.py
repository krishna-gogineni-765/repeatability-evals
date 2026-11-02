"""Estimate how much error pair subsampling adds versus scoring all pairs.

Synthetic by default; pass --runs-file (JSON: {question: [answer, ...]}) to use real runs
with ExactMatch-style token overlap (NgramJaccard) as a cheap stand-in for a judge.
"""
import argparse
import itertools
import json
import random
import statistics

from repeatability_evals import CycleSampler, NgramJaccard


def synthetic(n_q, n_runs, rng):
    out = []
    for _ in range(n_q):
        p = rng.choice([0.0, 0.1, 0.3, 0.6])  # per-question divergence
        base = [rng.random() for _ in range(n_runs)]
        out.append([0.0 if rng.random() > p else b for b in base])
    return out


def pair_score_synth(a, b):
    return 1.0 - min(1.0, abs(a - b) * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--runs-file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.runs_file:
        groups = list(json.load(open(a.runs_file)).values())
        sim = NgramJaccard(2)
        score = lambda x, y: sim(x, y)  # noqa: E731
    else:
        groups = synthetic(a.questions, a.runs, rng)
        score = pair_score_synth
    for budget in (None, 7, 8):
        errs = []
        for g in groups:
            n = len(g)
            full = statistics.mean(score(g[i], g[j]) for i, j in itertools.combinations(range(n), 2))
            ps = CycleSampler(budget=budget, seed=rng.randrange(10**6)).sample(n)
            sub = statistics.mean(score(g[i], g[j]) for i, j in ps)
            errs.append(sub - full)
        print(f"budget={budget or 'N'}: mean_abs_err={statistics.mean(map(abs, errs)):.3f} "
              f"bias={statistics.mean(errs):+.4f} max_abs_err={max(map(abs, errs)):.3f}")


if __name__ == "__main__":
    main()
