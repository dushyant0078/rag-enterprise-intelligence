# frontend/

Streamlit enterprise dashboard (Milestone 10) — a real client of the
FastAPI backend (`api/`, Milestone 9) over HTTP, never a shortcut around it
(see `frontend/api_client.py`).

Run locally: `make run-ui` (needs `make run-api` running first), then open
http://localhost:8501 and enter an API key in the sidebar.

## Pages (`pages/`)

1. **Chat Assistant** — conversational RAG (`POST /chat`), session memory,
   citations, selectable prompt template
2. **Document Upload** — ingest (`POST /upload`) then reindex
   (`POST /reindex`)
3. **Semantic Search** — retrieval only (`POST /search`), with an optional
   reranking toggle
4. **Summarization** — raw text or an ingested `doc_id`
   (`POST /summarize`)
5. **Compliance Analyzer** — the real `compliance_checking` template
   (verdict + reasoning), not a relabeled QA call
6. **Retrieval Explorer** — dense-only vs. reranked, side by side, for the
   same query
7. **Source Viewer** — full parsed content + metadata for one document
   (`GET /documents/{doc_id}`)
8. **Conversation History** — replay a past session's full turn history
   (`GET /sessions/{session_id}`)
9. **Evaluation Dashboard** — runs the Milestone 8 benchmark on demand
   (`POST /evaluate`)
10. **Admin Monitoring** — health, real Prometheus `/metrics` text, and
    configured components

See `docs/10_streamlit_dashboard.md` for the full writeup, including the API
extensions this milestone required (template selection, a rerank toggle,
and two new `GET` endpoints) and how the whole thing is tested end-to-end
against a real running backend, not mocks.
