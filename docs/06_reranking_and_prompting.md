# Milestone 6 — Reranking and Prompt Engineering

## What was built

### Reranking (`src/reranking/`)

| Backend | Status |
|---|---|
| `CrossEncoderReranker` (covers `bge_reranker` + `cross_encoder_ms_marco`) | Correct, complete `sentence-transformers.CrossEncoder` code — needs Hugging Face Hub access, same constraint as Milestone 4's embedding backends. Untestable here. |
| `CohereReranker` | Correct, complete `cohere` v7 SDK code — needs a Cohere API key and network access to Cohere's API. Untestable here. |
| `LexicalOverlapReranker` | **Real, fully offline, tested end-to-end.** |

`LexicalOverlapReranker` scores each (query, candidate) pair on three
signals: query-term recall, an exact-phrase-match bonus (weighted — a true
3-word match counts more than an incidental 2-word overlap), and a mild
length penalty. It's explicitly **not** presented as parity with a trained
cross-encoder — a cross-encoder learns relevance from millions of labeled
pairs via cross-attention; this is closed-form lexical scoring. What it
demonstrates for real is the mechanical value reranking adds beyond a
single first-pass score.

**A bug I found while building a fair test case, not after:** my first
version of the phrase-match bonus treated any 2-word or 3-word overlap as
equally strong evidence. Testing against a permuted-word-order pair (same
bag of words, scrambled) revealed the bug directly — an incidental 2-word
coincidence ("diligence wire") was scoring identically to a genuine 3-word
phrase match ("enhanced due diligence"). Fixed by weighting 3-grams above
2-grams, and kept the discovering test in the suite
(`test_trigram_match_scores_higher_than_incidental_bigram_match`).

### Prompting (`src/prompting/`)

- **`PromptTemplate`** (`schemas.py`) renders to a standard
  `[{"role": "system", ...}, {"role": "user", ...}]` chat-message list —
  the shape Milestone 7's LLM abstraction will consume directly.
- **`ContextBuilder`** assembles retrieved chunks into one citation-tagged,
  token-budgeted context string. Default budget (6000) is the exact value
  Milestone 2's `configs/config.yaml` declared for
  `context_builder.max_context_tokens` — this milestone is what makes that
  config value real.
- **Guardrail snippets** (`guardrails.py`) map directly to named risks
  from Milestone 1: `GROUNDING`/`NO_SPECULATION`/`INSUFFICIENT_EVIDENCE` →
  the hallucination-risk table, `CITATION_REQUIRED` → the citation-accuracy
  compliance risk, `INJECTION_RESISTANCE` → the prompt-injection security
  constraint. These are prompt-level mitigations; the Response Validator
  that actually *checks* compliance with them at runtime is Milestone 7's job.
- **All 8 templates** from the spec (QA, summarization, contract analysis,
  compliance checking, risk extraction, financial analysis, technical
  troubleshooting, research assistance), each with a task-specific system
  prompt, real guardrails, and at least one genuine few-shot example. The
  QA template's few-shot set includes both a grounded-answer example and an
  "insufficient evidence" example, so the fallback behavior is demonstrated,
  not just instructed.

### Cleanup along the way

Three retrieval modules (`multi_query.py`, `contextual_compression.py`,
`sparse_bm25.py`) had each independently grown an identical stopword list
and tokenization pattern. Consolidated into `src/utils/text.py`
(`STOPWORDS`, `tokenize()`, `content_words()`) — a "rule of three" cleanup,
verified against the full test suite before proceeding, not just assumed safe.

## What was actually run (not just written)

**`scripts/compare_reranking.py`**: BM25 first-pass retrieval (top-5) over
the real hybrid-chunked sample corpus, then `LexicalOverlapReranker`
rescoring, using the same 7 hand-written queries as Milestones 4 and 5.
Result, in `reports/milestone6_reranking_comparison.md`:

| | Hit@1 |
|---|---|
| Before reranking | 0.86 |
| After reranking | **1.0** |

The previously-failing query — *"what was the top risk factor reported last
quarter"* — moved from rank 3 to rank 1. **This is a different mechanism
than the Milestone 4/5 finding, and I diagnosed why rather than just
reporting the number:**

```
BM25 top-5 (raw scores):
  3.121  insurance FAQ chunk         <- ranked #1, incorrectly
  2.496  research abstract chunk
  1.945  financial CSV chunk (table) <- correct answer, ranked #3
  1.945  financial CSV chunk (raw)

After LexicalOverlapReranker:
  0.135  financial CSV chunk (table) <- now #1, correctly
  0.135  financial CSV chunk (raw)
  0.133  research abstract chunk
 -0.043  insurance FAQ chunk
```

The financial CSV's header literally contains the word **"quarter"** as a
standalone column name — a genuine, exact content-word match against the
query's "last quarter." The reranker's direct term-recall check picks this
up cleanly. BM25, by contrast, computes its score from **corpus-wide IDF
statistics over only 16 chunks** — too small a corpus for those statistics
to be stable, which is what let the insurance FAQ chunk outscore the
correct answer in the first pass. So the honest conclusion isn't "reranking
is smarter" in general — it's that **a reranker scoring each candidate
independently of global corpus statistics is more robust exactly when the
corpus is too small for those statistics to be reliable**, which is a real,
useful, and quite different lesson than Milestone 4/5's finding that no
amount of retrieval-side change could fix a tokenization gap. Both
diagnoses came from actually inspecting the raw scores, not from the
top-line Hit@1 number alone.

## Testing

- 52 new tests (7 reranking + 45 prompting), **221 tests total repo-wide,
  all passing, 82% coverage.**
- `black`/`isort`/`ruff`/`mypy` clean.
- The trigram-vs-bigram bug above was caught by a test written to *prove*
  the reranker adds value, not by a test written to check a known-correct
  answer — worth naming, since it's the more valuable kind of test to write.

## Known limitations / deferred work

- **`CrossEncoderReranker` and `CohereReranker` are correct, untested
  code** — same treatment as every Hugging-Hub/cloud-API-dependent
  backend since Milestone 4.
- **`LexicalOverlapReranker` shares TF-IDF/BM25's tokenization
  limitations** (no stemming, underscore-joined tokens treated as one
  word) — it isn't a fix for that class of gap, just a different, useful
  signal for a different class of problem (small-corpus statistical noise,
  word-order-blindness in bag-of-words scoring).
- **Prompt templates are not yet wired to an LLM** — that's Milestone 7.
  Guardrails are instructions in the prompt; nothing in this milestone
  checks whether the model actually followed them. That check (the
  Response Validator) is also Milestone 7.

---
*Next: Milestone 7 — conversational RAG pipeline.*
