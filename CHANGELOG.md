# Changelog

## 0.3.0

### Fixed
- **Dataset means no longer count single-run groups.** Groups with fewer than 2 items were scored
  1.0 and averaged in, inflating `evaluate_groups` results. They are now excluded and listed in
  `DatasetResult.degenerate`.
- **`acompare_versions(concurrency=N)` is a real global cap.** Previously each input and each metric
  had its own semaphore, so `concurrency=4` could issue ~24 simultaneous judge calls.
- **Judge prompts are filled in one pass.** A response containing `{item_b}` could pull the other
  response into its own slot. Responses are now also enclosed in tags with an instruction to
  treat them as data (`render_prompt` is public).
- **Score parsing ignores scale text.** `"Score (1-5): 4"` parsed as 1 and `"2 out of 5"` could
  parse as 5. JSON is now found even after prose containing braces, and rubric scores may be
  nested under `"scores"` or top level.
- `EmbeddingCosine` cache was not thread-safe under the async metrics (possible `KeyError`).
- `TotalAgreement` crashed on unhashable items (dicts, lists); they are compared as canonical JSON.
- `cycle_pairs` with a truncated budget now always covers `min(N, 2 * budget)` items (odd N missed
  one).
- `__version__` is read from package metadata (0.2.0 on PyPI reported 0.1.0).

### Added
- **Noise-aware per-input flags in `compare_versions`.** An input now counts as changed only if
  its drift exceeds `tolerance` *and* a permutation test on its pair scores (cross-version pairs
  less alike than within-version pairs) gives `p_value <= alpha` (default 0.2; `alpha=1.0`
  restores the old drift-only rule). On simulated noisy-but-unchanged systems (30% of runs
  differ, 4 runs per side) the share of inputs flagged by chance drops from ~21% to ~5%, and in
  `examples/compare_versions_demo.py` the flagged set now matches the 36 truly changed inputs
  exactly (previously 5 false positives). The dataset-level verdict is unchanged in method.
  `InputComparison.p_value` and CLI `--alpha` expose it.
- `on_error="skip"` on `PairwiseConsistency`, `evaluate_groups`, `aevaluate_groups`,
  `compare_versions`, `acompare_versions` and the CLI (`--on-error skip`): failing comparisons are
  dropped and reported instead of aborting a long run. `AllComparisonsFailed` when nothing is left.
- Judges: `retries` (default 2) for unparsable replies, `skip_identical` (default on: identical
  outputs score 1.0 with no model call) and an LRU `cache_size` (default 10,000). Symmetric async
  judging runs both orders concurrently.
- `Similarity.prepare` / `aprepare` hook, called with every item of a group; `EmbeddingCosine` uses
  it to embed a group in one batch, and accepts an async `aembed`.
- `call_limit(n)` context manager and `aevaluate_groups(max_calls=...)` for global rate limiting.
- `FunctionSimilarity` passes `context` to functions that accept it.
- CLI reads JSON Lines (`{"id", "output", "query"}` per line).
- HTML reports are now structured pages (tables, headings, light/dark) instead of preformatted
  markdown.

### Changed
- CLI exit codes: `0` no meaningful change, `3` inconclusive, `4` changed (were 0/1/2, which
  collided with Python and argparse error codes).
- Judge prompts wrap the input and responses in `<input>`, `<response_a>`, `<response_b>` tags.
  The rubric wording is unchanged.
