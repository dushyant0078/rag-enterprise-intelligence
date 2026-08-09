# Milestone 10 — Streamlit Enterprise Dashboard

## What was built

All 10 pages named in the spec, every one a real client of the FastAPI
backend (Milestone 9) via `frontend/api_client.py` — matching the
architecture diagram (User → Streamlit UI → FastAPI Gateway → ...), not a
shortcut that calls the pipeline directly.

| Page | Endpoint(s) | Notable behavior |
|---|---|---|
| Chat Assistant | `POST /chat` | Session id persists in `st.session_state` across turns — a real multi-turn conversation (with Milestone 7's query rewriting) happens across reruns |
| Document Upload | `POST /upload`, `POST /reindex` | Deliberately two steps, matching the API's own design |
| Semantic Search | `POST /search` | Reranking toggle |
| Summarization | `POST /summarize` | Raw text or an ingested `doc_id` |
| Compliance Analyzer | `POST /query` (`template=compliance_checking`) | The real Milestone 6 template — verdict + reasoning |
| Retrieval Explorer | `POST /search` ×2 | Live dense-vs-reranked comparison, interactive version of Milestone 6's `compare_reranking.py` |
| Source Viewer | `GET /documents/{doc_id}` | Full parsed content + metadata |
| Conversation History | `GET /sessions/{session_id}` | Session isolation enforced server-side |
| Evaluation Dashboard | `POST /evaluate` | Runs the real Milestone 8 benchmark on demand |
| Admin Monitoring | `GET /health`, `/metrics`, `/model-info` | Real Prometheus text, not a mock |

## API extensions this milestone required (not scope creep — each is what makes one page real)

Building pages against the *existing* Milestone 9 API surface would have
meant several pages either couldn't be built honestly or would have been
misleading reskins of endpoints that don't do what the page claims:

- **`ConversationalRAGPipeline.ask()` gained a per-call `prompt_template`
  override.** Without it, `/chat` and `/query` were hardcoded to the QA
  template — a "Compliance Analyzer" page built on that would just be a
  QA page with a different label, not actually using the real
  `compliance_checking` template (verdict + reasoning, real guardrails,
  real few-shot examples) Milestone 6 already built. `/query` and `/chat`
  now accept a `template` field (`Literal`-typed against the 7 templates
  that share `render(query=, context=)`'s signature — `summarization` is
  excluded, it has its own endpoint and a different signature).
- **`/search` gained a `rerank` boolean.** Without it, there was no way for
  a "Retrieval Explorer" page to show what Milestone 6 actually
  demonstrated (reranking changing the ordering, and why) — it could only
  show one fixed ranking.
- **`GET /documents/{doc_id}`** (new) — the Source Viewer's data source.
  RBAC-checked against the document's domain, not just its existence.
- **`GET /sessions/{session_id}`** (new) — the Conversation History page's
  data source. Isolation is enforced by `SessionMemoryStore` itself
  (Milestone 7), not re-implemented here: a scoped key gets a real 403
  trying to read another user's session.

## Testing: real, end-to-end, through the actual UI layer

Streamlit's `AppTest` (`streamlit.testing.v1`) runs the actual page
scripts and lets tests interact with real widgets (`text_input`, `button`,
`chat_input`, ...) and inspect what actually rendered — not a mock of
Streamlit. Pointing the page's `RAGAPIClient` at an in-process `TestClient`
wouldn't exercise the same code path a real browser session does, so
`tests/frontend/conftest.py` starts an **actual background `uvicorn`
server** on a free local port instead:

```
AppTest → real Streamlit page script → real HTTP → real running FastAPI
server → real pipeline (hybrid chunking, TF-IDF, FAISS, LexicalOverlap
Reranker, DeterministicTestLLM)
```

This is the same rigor as every other milestone's end-to-end verification
in this project, now extended through the UI layer. **18 tests**, every
one against the live server: a real multi-turn chat exchange, a real
upload → reindex → search round trip, a real dense-vs-reranked comparison,
a real session-history replay of an actual prior chat turn, and a real
full evaluation-benchmark run rendered as metrics.

Beyond `AppTest`, the app was also verified with a **real `streamlit run`
process** (not just in-process testing) — started, confirmed serving
`HTTP 200`, and stopped — the same additional verification layer
Milestone 9 did with a real `uvicorn` process for the API.

## Real issues found and fixed while building this milestone

- **mypy caught two real `Optional`-narrowing gaps**: `st.file_uploader()`
  returns `UploadedFile | None`, used before a null check even though the
  upload button was UI-disabled when no file was selected — `disabled=` is
  a UI hint, not a type guard, and isn't real safety either (a client could
  still send the request). Fixed with an explicit check. Separately,
  `/search`'s new `rerank` path called `state.rag_pipeline.reranker.rerank()`
  without accounting for `reranker: BaseReranker | None` — fixed by
  returning a clear `503` if no reranker is configured, rather than assuming
  one always will be.
- **`requests` was declared only in the `ingestion` optional dependency
  group**, but the frontend needs it unconditionally — moved to core
  dependencies rather than leaving the frontend accidentally dependent on
  an unrelated optional group being installed.
- **isort/ruff didn't recognize `frontend` as a first-party package**,
  producing import-sort churn identical to the isort/ruff disagreement
  fixed in Milestone 3 — fixed the same way, by adding `frontend` to both
  tools' `known-first-party` configuration rather than fighting it per-file.

## Testing summary

- 18 new tests (all real, live-server end-to-end), **428 tests total
  repo-wide, all passing, 85% coverage.**
- `black`/`isort`/`ruff`/`mypy` all clean across `src`, `api`, and
  `frontend` together.

## Known limitations / deferred work

- **`KNOWN_DOMAINS` in `frontend/config.py` is a hardcoded list** matching
  the sample corpus's domains, not fetched from the backend — a documented
  simplification; a `GET /domains` endpoint would be the natural real
  fix, deferred since every page's domain field also accepts free text.
- **No user-facing distinction between "not grounded because off-topic"
  and "not grounded because of low confidence on an otherwise-correct
  retrieval"** (the calibration finding from Milestones 7-8) — the UI
  shows the same "⚠️ Not grounded" badge either way; a real production UI
  would likely want to surface that nuance.
- **The dashboard has no its own authentication beyond the API key
  typed into the sidebar** — reasonable for this project's scope; a real
  deployment would put the dashboard itself behind SSO and let the backend
  API key be provisioned per authenticated user, not typed in by hand.

---
*Next: Milestone 11 — monitoring and observability.*
