# Milestone 9 — FastAPI Backend

## What was built

### Security layer (`src/security/`)

The spec's "LLM SAFETY & SECURITY" section has no dedicated milestone of its
own -- its items (API authentication, RBAC, rate limiting, PII masking,
prompt injection detection, audit logging) are API-Gateway-level concerns,
so they landed here, in the folder Milestone 2 already reserved for exactly
this. Consistent with this project's pattern for every heuristic since
Milestone 5: real, deterministic, testable, and honestly scoped.

- **`ApiKeyRegistry`** -- API key -> identity + `allowed_domains` (RBAC).
  In-memory, not a production credential store (documented as such); the
  *enforcement* is real -- `MetadataAwareRetriever`'s RBAC pattern from
  Milestone 5, now applied at the API boundary via a shared
  `resolve_domain_filters()` helper.
- **`TokenBucketRateLimiter`** -- real token-bucket algorithm, deterministic
  and testable with an injectable clock.
- **`mask_pii`** -- regex-based masking (SSN, credit card, email, phone) for
  audit logs only, never applied to the text the pipeline actually processes.
- **`detect_injection_attempt`** -- a heuristic first layer at the gateway,
  complementing (not replacing) the `INJECTION_RESISTANCE` prompt guardrail
  from Milestone 6.
- **`log_request`** -- structured, PII-masked audit logging via the same
  JSON logger the rest of the pipeline uses.

### FastAPI application (`api/`)

Every endpoint from the spec, all wired to real Milestone 3-8 components,
not stubs:

| Endpoint | What it does |
|---|---|
| `GET /` | API info |
| `GET /health` | Component status |
| `GET /metrics` | **Real Prometheus text exposition format** -- scrapeable today, not a mock |
| `GET /model-info` | Introspects the configured embedding/LLM/vector-store/reranker |
| `POST /chat` | Conversational RAG -- wraps `ConversationalRAGPipeline.ask()` |
| `POST /query` | Stateless single-turn RAG |
| `POST /search` | Retrieval only, no generation |
| `POST /summarize` | Summarization template + LLM, by raw text or `doc_id` |
| `POST /upload` | Real multipart file upload -> `IngestionPipeline` (Milestone 3) |
| `POST /reindex` | Rebuilds the chunk/embedding/vector index from `data/processed/` |
| `POST /evaluate` | Runs the Milestone 8 evaluation framework on demand |
| `GET /docs`, `/openapi.json` | FastAPI's standard auto-generated Swagger UI / OpenAPI schema -- nothing built separately |

Every resource (vector store, embedding model, LLM, pipeline) is built once
at startup via a `lifespan` context manager and stored on `app.state` -- see
`api/state.py`. Data paths are environment-variable-overridable
(`RAG_PROCESSED_DIR`, etc.), which is both a legitimate production config
surface and what lets the test suite run against fully isolated tmp
directories instead of the real project's `data/`.

### Shared pipeline factory (`src/memory/pipeline_factory.py`)

The "load corpus -> chunk -> embed -> index -> build a retriever" sequence
had been written independently in each of Milestones 5-8's demo scripts.
The API is a fifth consumer -- past the point where copy-pasting it again
was reasonable -- so it's consolidated into one factory the API uses. The
milestone scripts themselves were left as-is (each is a snapshot of that
milestone's own demo), not retrofitted.

## Real bugs found by actually running the system

Every one of these was caught by exercising the real API end-to-end
(`TestClient`, and separately a real `uvicorn` process hit with `curl`),
not by unit tests against convenient, hand-picked fixtures:

1. **`/reindex` crashed with a FAISS dimension mismatch.** Rebuilding the
   index reused a stale persisted store from the previous corpus, whose
   dimensionality doesn't match TF-IDF's freshly-fit vocabulary once the
   document set changes. Fixed by clearing the persisted index directory on
   every full rebuild (`build_index()` is documented as always-full-rebuild,
   never incremental, for exactly this reason), with a regression test that
   reproduces the exact "upload, then reindex" scenario.
2. **`/summarize` returned garbled output.** `DeterministicTestLLM`'s
   context-block parser (built and tested against the QA template in
   Milestone 7) didn't recognize the summarization template's different
   closing structure, so it folded trailing prompt instructions into the
   "answer." Fixed the boundary detection generically (a short trailing
   `"Label:"` line, matching how every one of this project's own templates
   closes), then caught and fixed a **regression the first fix caused**
   against the QA template's inline `"Question: {query}"` format (which has
   content after the colon on the same line, not a label alone). Both
   directions are now regression tests.
3. **A domain-filtering test failed with zero results, not wrong ones.**
   Traced to my own test fixtures using plain doc ids (`"hr-doc"`) instead
   of the `domain__slug__hash` convention Milestone 3's real ingestion
   pipeline always produces -- which is what domain extraction actually
   parses, not which folder a file happens to be saved under. Fixed the
   fixtures and added a regression test directly on
   `load_processed_documents()`, the function whose implicit contract broke
   silently rather than loudly.

## Testing

- **396 tests passing repo-wide**, 46 of them real HTTP tests against the
  live app (`tests/api/`): authentication rejection, a genuine RBAC case
  (an HR-scoped key correctly denied a grounded banking answer, and
  correctly granted one for HR content), rate-limit 429s, prompt-injection
  400s, request validation, and a full `upload -> reindex -> query` round
  trip that confirms a newly uploaded document is actually retrievable and
  correctly cited afterward.
- 16 dedicated unit tests for `src/security/`, plus indirect coverage
  through the API tests.
- `black`/`isort`/`ruff`/`mypy` all clean. mypy caught 7 real type issues
  (`Optional` narrowing gaps on `body.doc_id`/`file.filename`, and a
  `dict[str, str]` vs. `dict[str, str | list[str]]` mismatch in the RBAC
  filter-construction logic) -- fixed by real narrowing, and the RBAC fix
  became an opportunity to consolidate three copies of identical filter
  logic (chat/query/search) into one shared, properly-typed
  `resolve_domain_filters()` helper rather than patch each copy separately.
  Ruff needed one legitimate FastAPI-specific carve-out
  (`B008` for `Depends()`-as-default, which is FastAPI's standard,
  documented pattern, not a real code smell).

## Known limitations / deferred work

- **`ApiKeyRegistry` is in-memory, not a production identity store** -- a
  real deployment needs a secrets manager or identity provider behind this;
  the enforcement hook (RBAC intersection at the retrieval boundary) is
  real, the credential *storage* is a documented starting point.
- **`/reindex` is always a full corpus rebuild**, not a true incremental
  per-domain update -- accepted as a reasonable v1 scope given this
  project's corpus sizes; a documented extension, not an oversight.
- **`detect_injection_attempt` is a fixed-phrase heuristic** -- a floor, not
  a ceiling, same honesty as every other heuristic in this project since
  Milestone 5.
- **The full OpenTelemetry tracing/collector pipeline is Milestone 11's
  job** -- `/metrics`'s Prometheus counters are the request-counting layer
  that would feed it, not a replacement for it.

---
*Next: Milestone 10 -- Streamlit enterprise dashboard.*
