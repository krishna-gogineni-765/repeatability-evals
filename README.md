# consistency-evals

Composable tools for measuring **how consistent an LLM system is with itself**: the same
input run several times, or an input and its paraphrases, and how much the outputs agree.

Most eval libraries score one run for quality. A system can average 90% on quality and still
give a customer a different answer every time they ask. This package measures that spread.

It is deliberately **unaware of what you are comparing**. It has no notion of "answers",
"retrieval" or "agents": it compares *items*. You choose the items and the similarity:

| You have N of ...                    | Use a similarity such as                         |
|--------------------------------------|--------------------------------------------------|
| final answers (free text)            | `LLMJudge`, `EmbeddingCosine`, `RougeL`          |
| retrieved chunk IDs                  | `JaccardSimilarity`                              |
| retrieved chunk text                 | `NgramJaccard` (robust to re-chunking)           |
| ranked result lists                  | `RankBiasedOverlap`                              |
| agent tool-call traces               | `EditSimilarity` / `ExactMatch` on a trace string|
| anything else                        | subclass `Similarity`, or pass a plain function  |

Consistency is **not** correctness: a system that is consistently wrong scores 1.0. Pair this
with an accuracy metric.

## Install

```bash
pip install consistency-evals   # pure standard library, no required dependencies
```

## Quickstart

```python
from consistency_evals import (
    PairwiseConsistency, JaccardSimilarity, NgramJaccard, LLMJudge, evaluate_groups,
)

# 1. Retrieval: did the 5 runs fetch the same chunks?
ids = PairwiseConsistency(JaccardSimilarity())
print(ids.evaluate([["a", "b", "c"], ["a", "b", "d"], ["a", "b", "c"]]).score)

# 2. Same evidence, different chunk boundaries?
text = PairwiseConsistency(NgramJaccard(n=3))
print(text.evaluate([["backup jobs run nightly"], ["backup jobs", "run nightly"]]).score)

# 3. Answers: an LLM judge scores semantic equivalence. Bring any `str -> str` function.
def complete(prompt: str) -> str: ...            # your model call
answers = PairwiseConsistency(LLMJudge(complete), agreement_threshold=0.75)
r = answers.evaluate(["Retention is 30 days.", "30 days.", "Retention is 90 days."],
                     context={"query": "How long are backups retained?"})
print(r.score, r.details["agreement_rate"], [p.reason for p in r.pairs])

# 4. A whole dataset: one group per question, with a bootstrap confidence interval.
report = evaluate_groups({"q1": runs_q1, "q2": runs_q2}, ids)
print(report.score, report.ci, report.worst(3))
```

## Design

Three small pieces, each replaceable:

* **`Similarity`**: compare two items, return a score in `[0, 1]` (optionally with a raw value
  and a rationale). Built in: set overlap (`Jaccard`, `Dice`, `OverlapCoefficient`), text
  (`ExactMatch`, `NgramJaccard`, `RougeL`, `EditSimilarity`), ranking (`RankBiasedOverlap`),
  embeddings (`EmbeddingCosine`, bring your own embed function), and `LLMJudge`.
* **`PairSampler`**: which pairs of the N items to compare (see below).
* **`ConsistencyMetric`**: combines them. `PairwiseConsistency` averages similarity over
  sampled pairs; `ClusterConsistency` groups items into "same answer" clusters and reports the
  largest cluster's share plus the entropy of the partition; `TotalAgreement` is the
  all-N-identical rate.

Everything has an async twin (`aevaluate`, `aevaluate_groups`) with bounded concurrency, which
matters when the similarity is an LLM call.

### Sampling instead of all pairs

All pairs cost N(N-1)/2 comparisons: 10 for 5 runs, 45 for 10. The default `CycleSampler`
compares a random cycle through the runs, so N pairs, and every run is compared at least twice.
It is seeded and reproducible. `AllPairs()` gives the exact value; `RandomPairs(k)` and
`AnchorPairs()` (every run against one reference) are also available. `benchmarks/sampling_error.py`
measures the error this introduces on synthetic data so you can see the trade-off for your N.

### The LLM judge

`LLMJudge` takes any completion function, so it works with any provider or a local model. The
default rubric (1-5, normalised to `[0, 1]`) is in `consistency_evals.prompts`. `symmetric=True`
judges both orderings to cancel position bias at twice the cost. **Validate any judge against
human ratings on your own data before relying on it**; agreement depends on the domain and the
judge model.

## Writing your own similarity

```python
from consistency_evals import Similarity, PairwiseConsistency

class SameFirstSentence(Similarity):
    name = "same_first_sentence"
    def compute(self, a, b, *, context=None):
        return float(a.split(".")[0] == b.split(".")[0])

PairwiseConsistency(SameFirstSentence())
```

or just `PairwiseConsistency(lambda a, b: ...)`.

## Status

Early (0.1). The API may change before 1.0.

## License

MIT
