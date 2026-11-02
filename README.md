# repeatability-evals

**Consistency evals for non-deterministic systems: LLM generation, agentic retrieval, agent workflows.**

Most evals ask *"is the output correct?"* and need gold labels to answer. This library asks
two questions that are extremely important in enterprise settings and need **no gold labels**:

1. **Is my system consistent/repeatable with itself?** Same input, N runs: do the answers, the retrieved
   documents, the tool-call traces agree?
2. **Did anything user facing change when I changed something in the backend?** Model upgrade, prompt edit, new index,
   framework bump: which inputs behave differently, *beyond the system's own run-to-run noise*,
   and *how* (facts? numbers? format? refusals?).

## Why it matters

| If you are... | You can... |
|---|---|
| **a platform or model provider** shipping an update | run it offline on a regression set and see which customer-style workflows would shift before you release |
| **an enterprise on a vendor's LLM or agent platform** | run it yourself, on your own data and cluster, to decide whether an upstream upgrade is safe for the workflows you built. Nothing leaves your environment, and `include_text=False` keeps raw text out of reports |
| **building an agent, RAG or harness product** | track whether retrieval and answers stay stable across runs, and catch silent drift from non-deterministic infrastructure |
| **without labelled data** | still get a signal: consistency needs only outputs, not answers |


## Usage pattern 1: See what changed

```python
from repeatability_evals import compare_versions, RubricJudge

# {input_id: [output of run 1, run 2, run 3, ...]} for each version. No labels.
report = compare_versions(v1_runs, v2_runs, RubricJudge(my_llm), names=("v1", "v2"))
print(report.verdict)            # no_meaningful_change | changed | inconclusive
report.save("report.md")         # also .html / .json
```

```
# v1 vs v2: consistency report
CHANGED: the candidate's outputs differ from the baseline's by more than run-to-run noise explains.
120 inputs compared; 30% changed beyond the noise floor.

| v1 vs itself (noise floor)               | 0.925 |
| v2 vs itself                             | 0.948 |
| v1 vs v2                                 | 0.850 |
| drift (expected - cross; 0 = no change)  | +0.087 (95% CI +0.060 to +0.116) |

## What changed
| format            | +0.200 |
| specifics         | +0.115 |
| supporting_detail | +0.100 |
| core_facts        | +0.057 |
Changed inputs by main driver: 21 mainly in `format`, 15 mainly in `specifics`.

### `q000`: drift +0.62 (cross 0.38 vs expected 1.00, p = 0.001)
- Main driver: `specifics`
- Judge: Retention period differs.
- v1: Retention for gold is 30 days.
- v2: - The retention period of gold is 90 days. - Applies to all sites.
```

*(Real output of `examples/compare_versions_demo.py`, which uses simulated runs and a scripted
stand-in for the LLM judge so it runs without an API key.)*

The key idea is the **noise floor**. A non-deterministic system never matches itself perfectly,
so comparing v2 to v1 naively overstates change. We measure how much v1 differs from *itself*
and v2 from *itself*, and report only the **extra** disagreement across versions (`drift`).
Two equally noisy systems show no drift; a real change does. Each input is also screened with a
permutation test (are its cross-version pairs less alike than its within-version pairs?), so a
noisy system does not get a long list of inputs flagged by chance. In the demo above, 36 inputs
were changed by construction and exactly those 36 are flagged. The report also says *what kind*
of change it is, since format drift and changed numbers deserve different decisions. Use 3 or
more runs per side, 5 if the system is noisy.

From the shell, with no code and no model (offline n-gram similarity), or with your own LLM:

```bash
pip install repeatability-evals
repeatability-evals compare v1.json v2.json --out report.md
repeatability-evals compare v1.json v2.json --judge mypkg.llm:complete --rubric --out report.html
```

Input is JSON, `{"input id": ["run 1", "run 2", ...]}`, or JSON Lines with one run per line,
`{"id": "...", "output": "...", "query": "..."}`, which is the shape most request logs already
have. The exit code is `0` (no meaningful change), `3` (inconclusive) or `4` (changed), so it can
gate a CI pipeline.

## Usage pattern 2: Measure self-consistency

```python
from repeatability_evals import PairwiseConsistency, JaccardSimilarity, NgramJaccard, RubricJudge

PairwiseConsistency(JaccardSimilarity()).evaluate(chunk_ids_per_run)                 # retrieved IDs
PairwiseConsistency(NgramJaccard(3)).evaluate(chunk_texts_per_run)                   # robust to re-chunking
PairwiseConsistency(RubricJudge(my_llm)).evaluate(answers, context={"query": q})     # answers
```

Over a dataset, `evaluate_groups({input_id: runs, ...}, metric)` returns the mean, a bootstrap
confidence interval and the least consistent inputs.

## Design

The library never knows whether you are comparing answers, chunk IDs or agent traces. It
compares *items*; you pick the similarity metric that makes sense for what you are comparing.

* **`Similarity`** scores two items in `[0, 1]`: sets (`JaccardSimilarity`, `DiceSimilarity`,
  `OverlapCoefficient`), text (`ExactMatch`, `NgramJaccard`, `RougeL`, `EditSimilarity`),
  rankings (`RankBiasedOverlap`), embeddings (`EmbeddingCosine`, bring your own embed
  function) and LLM judges (`RubricJudge`, `LLMJudge`). Subclass it, or pass a plain function.
* **`PairSampler`** picks which of the N(N-1)/2 pairs to compare. The default `CycleSampler`
  costs N comparisons and touches every run twice; `CrossSampler` does the same across versions.
* **Metrics**: `PairwiseConsistency` (mean similarity), `ClusterConsistency` (largest
  "same answer" cluster, plus entropy), `TotalAgreement` (all N identical) and `compare_versions`.

Everything has an async twin (`aevaluate`, `acompare_versions`) with bounded concurrency. The
core has **no dependencies**; you bring the model, as any `str -> str` function.

### The judge

`RubricJudge` scores five dimensions of agreement in one call: `core_facts`, `specifics`,
`supporting_detail`, `stance` (answer vs hedge vs refuse) and `format`. The headline is a
weighted mean (core facts count most) and the per-dimension means explain *what* varies. Pass
your own `Dimension`s to change them. `LLMJudge` is the single 1-5 semantic-equivalence score.
`symmetric=True` judges both orders to cancel position bias. **Validate any judge against human
ratings on your own data before relying on it.**

### Running it at scale

LLM-judged evaluations over thousands of inputs should not fall over on one bad reply, and
should not pay for comparisons whose answer is obvious:

* **Failures**: judges re-ask when a reply has no parsable score (`retries=2`). Pass
  `on_error="skip"` to metrics, `evaluate_groups` or `compare_versions` to drop a comparison that
  still fails (timeout, API error) and keep going; skipped pairs and inputs are reported.
* **Cost**: identical outputs, which deterministic systems produce constantly, score 1.0 without a
  model call (`skip_identical=True`), and repeated comparisons are cached. `EmbeddingCosine`
  embeds each group in one batch.
* **Rate limits**: `concurrency` on `acompare_versions` (and `max_calls` on `aevaluate_groups`)
  caps simultaneous judge calls across the whole run.
* **Prompt safety**: outputs are inserted into the judge prompt in one pass, inside tags, and the
  judge is told to treat them as data, so placeholder-like text in a response cannot change the
  prompt's structure. (No prompt fully rules out injection; review judge reasons on outliers.)

### Sampling cost

All pairs for 10 runs is 45 judge calls; the default sampler uses 10. On synthetic data the
per-input error versus all pairs was about 0.03 with no bias (`benchmarks/sampling_error.py`);
check it on your data, or use `AllPairs()` for the exact value.

### Writing your own similarity

```python
from repeatability_evals import Similarity, PairwiseConsistency

class SameFirstSentence(Similarity):
    name = "same_first_sentence"
    def compute(self, a, b, *, context=None):
        return float(a.split(".")[0] == b.split(".")[0])

PairwiseConsistency(SameFirstSentence())      # or PairwiseConsistency(lambda a, b: ...)
```

