# Milestone 8 — Evaluation Framework

## What was built

This milestone formalizes what Milestones 4-7's individual comparison
scripts each did ad hoc (build an index, run hand-written queries, print a
hit rate) into one reusable evaluation framework computing every metric
named in the spec.

### Retrieval metrics (`src/evaluation/retrieval_metrics.py`) — exact, no LLM

Precision@K, Recall@K, MRR, nDCG@K, Context Precision, and Context Recall,
computed against binary relevance ground truth. These need no LLM, no
embedding model, nothing but the retrieved doc_id list and a labeled
benchmark — real, deterministic formulas, verified against hand-computed
values before being trusted in the framework.

**A real bug caught during that verification:** an early version of nDCG
normalized by `len(relevant_ids)` (the count of relevant *domain prefixes*
in the ground truth) rather than the actual number of relevant items found
in the retrieved set. Since one relevant prefix can match multiple
retrieved chunks, this let nDCG exceed 1.0 — mathematically impossible for
a metric that's supposed to be bounded in `[0, 1]`. Fixed by normalizing
against the actual hit count within the retrieved top-K, and kept as a
regression test (`test_ndcg_never_exceeds_one_with_multiple_matches`).

### Generation metrics (`src/evaluation/generation_metrics.py`) — one exact, three honest proxies

- **Citation Accuracy** is exact — a direct count against Milestone 7's
  `CitationGenerator` resolution logic, no LLM needed.
- **Hallucination Rate**, **Faithfulness**, and **Answer Correctness** are
  RAGAS/DeepEval concepts that, in standard form, use an LLM judge for
  entailment and semantic-similarity checking. No LLM is reachable
  end-to-end in this sandbox with valid credentials (see below), so these
  are real, deterministic *lexical* proxies — genuinely informative at this
  project's scale, explicitly not a claim of parity with an LLM judge.
  Faithfulness/hallucination rate check per-sentence term overlap against
  the retrieved chunks; answer correctness is token-overlap F1 against the
  benchmark's expected answer.

### Custom benchmark dataset (`src/evaluation/benchmark_dataset.py`)

The 7 hand-written queries used informally across Milestones 4-7 are now a
real, labeled, JSON-persistable dataset: each has a verified
`expected_answer` (read from the actual sample documents, not guessed) and
ground-truth `relevant_doc_ids` for scoring retrieval.

### `RAGEvaluator` (`src/evaluation/evaluator.py`)

Runs a `BenchmarkDataset` through a real `ConversationalRAGPipeline`,
computing every metric above per query and in aggregate, plus response
latency (p50/p95/mean — real, exact statistics over `LLMResponse.latency_ms`
already collected since Milestone 7). Each benchmark query gets a **fresh
session**, not a shared one — benchmark items are independent trials, not
one running conversation, and reusing a session would let query rewriting
or accumulated history leak between unrelated items.

### RAGAS and DeepEval integration

| Library | Status |
|---|---|
| DeepEval | **Imports cleanly; `LLMTestCase` construction verified working with no LLM.** Metric *construction* itself (not just `.measure()`) requires a configured judge — confirmed by actually calling `FaithfulnessMetric()` and getting `deepeval.errors.DeepEvalError: OpenAI API key is not configured`, a more precise finding than assumed from documentation alone. |
| RAGAS | **Cannot be imported in this sandbox at all — a genuine, reproducible upstream packaging conflict**, not a network restriction. `ragas.llms.base` unconditionally imports `langchain_community.chat_models.vertexai`, which doesn't exist in the `langchain-community` version ragas's own dependency resolution installs. Downgrading `langchain-community` to restore that module breaks `langchain-openai`'s requirement on a newer `langchain-core`. A real, current three-way version conflict in the PyPI ecosystem, confirmed via the actual error messages (and, fittingly, `langchain-community`'s own deprecation warning surfaced while writing the regression test that documents this). `ragas_integration.py`'s functions are written directly from RAGAS's documented API rather than verified against a working import — flagged as such in the module docstring, not passed off as tested. |

## What was actually run (not just written)

**`scripts/run_evaluation_framework.py`** runs the complete framework
against the real sample corpus: hybrid chunking, TF-IDF embeddings, FAISS,
`LexicalOverlapReranker`, `DeterministicTestLLM` — the real, testable stack
established across Milestones 4-7. Full results in
`reports/milestone8_evaluation_report.md`:

| Retrieval | Value | | Generation | Value |
|---|---|---|---|---|
| Precision@3 | 0.667 | | Citation Accuracy | 1.000 |
| Recall@3 | 1.000 | | Hallucination Rate | 0.433 |
| MRR | 1.000 | | Faithfulness | 0.567 |
| nDCG@3 | 0.989 | | Answer Correctness | 0.257 |
| Context Precision | 0.976 | | Grounded Rate | 0.714 |
| Context Recall | 1.000 | | | |

Retrieval is strong across the board — MRR of 1.0 means the correct
document was always the top hit. Generation metrics are more modest, and
that gap is itself the honest story this framework is supposed to surface,
not something to explain away:

- **Answer Correctness (0.257) is low mainly because it's a strict lexical
  F1 against a full reference answer**, and `DeterministicTestLLM` returns
  an extracted source sentence, not a synthesized answer matching the
  reference's exact phrasing — expected, given Milestone 7 already
  established this backend is extractive, not generative.
- **2 of 7 queries were "ungrounded"** (correctly refused per the strict
  `min_confidence: 0.7` policy) — and both trace to the same root cause
  Milestone 7 diagnosed: the reranker/retrieval confidence scores this
  offline stack produces sit on a different, generally lower scale than a
  dense embedding model's would, so a calibration threshold tuned with
  dense embeddings in mind is conservative here. One of these two queries
  (*"what was the top risk factor reported last quarter"*) is new to this
  benchmark and wasn't tested in Milestone 7's demo (which used a
  deliberately off-topic query as its 7th item instead) — this is a new,
  honest finding, not a regression, and it strengthens rather than
  contradicts Milestone 7's diagnosis by reproducing the same mechanism on
  a different query.

## Testing

- 44 new tests (retrieval metrics: 16, generation metrics: 12, latency +
  benchmark: 9, evaluator: 4, DeepEval integration: 2, RAGAS diagnosis: 1)
  — **331 tests total repo-wide, all passing, 84% coverage.**
- The nDCG bug above is represented as a named regression test, not just a
  passing assertion for the fixed formula.
- The RAGAS import-failure test is deliberately written to *pass now and
  fail later* — if a future release fixes the conflict, the test breaks,
  which is the correct signal to replace it with a real integration test
  rather than patch it to keep passing.
- `black`/`isort`/`ruff`/`mypy` all clean — mypy caught one more real issue
  (list invariance on RAGAS's `EvaluationDataset.samples` parameter,
  resolved against ragas's actual type stubs even though the runtime import
  fails, since mypy doesn't execute the broken import chain) on top of the
  nDCG bug found through testing.

## Known limitations / deferred work

- **Faithfulness, hallucination rate, and answer correctness are lexical
  proxies, not LLM-judge metrics** — real and useful at this scale,
  explicitly not a claim of matching RAGAS/DeepEval's standard
  entailment-based versions. `deepeval_integration.py` is the verified
  upgrade path once a judge model with credentials is available.
- **RAGAS integration is unverified** due to the documented upstream
  conflict — worth revisiting once ragas or langchain-community ships a fix,
  or with a project-specific pinned dependency set that resolves it.
- **The 0.7 grounding threshold's calibration gap is now confirmed twice**
  (Milestones 7 and 8) against this project's offline lexical stack — a
  real embedding model and reranker would be the natural next check, once
  one is reachable end-to-end.
- **Latency is measured only end-to-end**, not broken into a real
  per-stage (retrieval vs. generation) split — `retrieval_ms` is currently
  derived as `total - generation`, an approximation rather than a directly
  measured figure; real per-stage instrumentation is a natural extension
  for Milestone 11's monitoring work.

---
*Next: Milestone 9 — FastAPI backend.*
