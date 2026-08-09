# Milestone 11 — Monitoring and Observability

## What was built

Every item in the spec's Monitoring & Observability section, in
`src/monitoring/`, wired into the API (new `POST /feedback` and
`GET /monitoring/summary` endpoints, and every `/chat`/`/query` response now
feeds the aggregator) and the frontend (Admin Monitoring page shows the
live summary; Chat Assistant has real thumbs up/down buttons).

| Item | Module | Status |
|---|---|---|
| OpenTelemetry | `tracing.py` + `rag_pipeline.py` | **Real, verified spans** — every pipeline stage, plus FastAPI HTTP-level auto-instrumentation |
| Structured logging | (existing, `src/utils/logging_config.py`) | Already real since Milestone 3; unchanged here |
| Token usage tracking | `token_tracking.py` | Real aggregation over `LLMResponse` fields already produced since Milestone 7 |
| Cost monitoring | `cost_tracking.py` | Real math over real (caveated, point-in-time) pricing tables |
| Latency monitoring | `rag_pipeline.py` | **Real per-stage latency**, closing a gap Milestones 8-9 explicitly documented as missing |
| Retrieval performance monitoring | `retrieval_monitoring.py` | Real live statistics — distinct from Milestone 8's benchmark-based evaluation |
| Hallucination alerts | `hallucination_alerts.py` | Real rolling-window threshold alerting, verified to actually fire |
| Data drift monitoring | `drift_monitoring.py` | **Real two-sample KS test** (`scipy.stats.ks_2samp`), verified against genuinely different distributions |
| User feedback analytics | `feedback.py` | Real disk-persisted store, wired end-to-end from UI to API to aggregate |
| LangSmith tracing | `langsmith_integration.py` | Import-verified; untestable here — `api.smith.langchain.com` confirmed blocked at the sandbox's egress proxy |

## Closing a documented gap: real per-stage latency

Milestones 8 and 9 both explicitly flagged the same limitation: "only
overall (retrieval + generation combined) latency is measured end-to-end
... a natural extension for Milestone 11's monitoring instrumentation."
`ConversationalRAGPipeline.ask()` now wraps every stage — query rewrite,
retrieval, rerank, context build, generation, validation — in a real
`time.monotonic()` measurement, recorded in
`ValidatedAnswer.metadata["stage_latencies_ms"]`. Verified directly:

```
stage_latencies_ms: {'query_rewrite_ms': 0.031, 'retrieval_ms': 0.777,
'rerank_ms': 0.709, 'context_build_ms': 0.013, 'generation_ms': 0.169,
'validation_ms': 0.051, 'total_ms': 1.750}
```

## Real OpenTelemetry, not a stub

Each pipeline stage above is *also* wrapped in a real OTel span, via a
tracer injected into `ConversationalRAGPipeline.__init__` (defaulting to
whatever global provider is configured — a harmless no-op if tracing was
never set up, by design of the OTel API). Verified with a fully isolated
`InMemorySpanExporter` against a real pipeline run:

```
6 spans captured: conversation_manager.prepare_turn, retriever.retrieve
(chunks_retrieved=10), reranker.rerank, context_builder.build, llm.generate
(model=...), response_validator.validate (is_grounded=True)
```

`api/main.py` also wires in `FastAPIInstrumentor.instrument_app(app)` for
HTTP-level spans. Confirmed both layers compose correctly in one trace —
an isolated exporter attached before the app started captured **15 spans**
for two requests: `GET /health`, then `POST /query` with all 6 pipeline
stages nested inside it, then the response spans. `RAG_TRACE_CONSOLE=true`
prints every span to stdout for local debugging; `OTEL_EXPORTER_OTLP_ENDPOINT`
(already a Milestone 2 placeholder in `.env.example`) is now actually read
and wired to a real `OTLPSpanExporter` if set — that endpoint reachability
itself is outside this sandbox's network allowlist, so the export path is
correct code, not verified end-to-end to a real collector.

## Real statistics for drift, not a heuristic

`DriftMonitor` uses `scipy.stats.ks_2samp` — the standard tool for
comparing two empirical distributions — rather than a bespoke mean/std
comparison. Verified against two clearly distinguishable synthetic cases:
a same-distribution comparison correctly returns `drifted=False` (p≈0.61),
and a genuinely shifted distribution (mean 0.8 → 0.4) correctly returns
`drifted=True` (p≈1e-40). Refuses rather than returns an unreliable result
below 10 samples in either window.

## Real bug found and fixed: test isolation for feedback

Wiring `FeedbackStore` in revealed a real gap: the existing test fixtures
(`tests/api/conftest.py`, `tests/frontend/conftest.py`) set every
`RAG_*_DIR` env var *except* the newly-added `RAG_FEEDBACK_DIR`, so
`FeedbackStore` silently fell back to the real project's `data/feedback/`
directory. Every test run was polluting that real folder, and a later
test's feedback count silently included earlier tests' leftover records —
caught by an exact-count assertion (`total == 1`), not a loose
"at least one" check, which would have passed anyway despite the leak.
Fixed in both fixtures, added `data/feedback/*` to `.gitignore`, cleaned
up the polluted files, and kept a named regression test explaining the
failure mode for future-me.

## Testing

- **39 new `src/monitoring/` unit tests** — token/cost math, retrieval
  monitor windowing, hallucination alert threshold/severity/window-rollover
  behavior, drift detection against real distinguishable distributions,
  feedback persistence and aggregation, isolated-provider OTel span capture
  (including one real end-to-end pipeline-run span check), and LangSmith's
  real no-network-needed surface (decorator behavior, config detection).
- **9 new API tests** (`/feedback`, `/monitoring/summary`, including the
  test-isolation regression above) and **3 new frontend tests** (a real
  thumbs-up submission through the actual Chat Assistant UI, and the Admin
  Monitoring page reflecting a real chat + feedback round trip through the
  live server).
- **477 tests total repo-wide, all passing, 86% coverage.**
- `black`/`isort`/`ruff`/`mypy` all clean across `src`, `api`, and
  `frontend` — ruff caught one real unused-variable slip in a test
  (`tracer_b` constructed to prove provider isolation but never
  referenced), fixed by keeping the construction and dropping the
  unused binding.

## Known limitations / deferred work

- **Cost pricing is a point-in-time snapshot**, explicitly documented as
  such in `cost_tracking.py` — a real deployment should verify current
  rates against each provider's pricing page, not trust this table
  indefinitely. An unknown model returns cost `0.0` with
  `pricing_known=False`, never a silently wrong number.
- **`DriftMonitor` has no baseline-management workflow** (when to
  (re)capture a baseline, how long it should remain valid) — the class
  itself is real and correct; deciding an operational policy around it is
  a deployment decision, not something this milestone prescribes.
- **LangSmith tracing is genuinely untested** beyond import/decorator
  verification, for the same network reason as RAGAS (Milestone 8) and
  every cloud LLM/vector-store backend before it.
- **The monitoring aggregator is in-process, single-instance state** — a
  real multi-replica deployment would need this backed by a shared store
  (Redis, a time-series DB) rather than living in one API process's
  memory; `MonitoringAggregator`'s interface doesn't preclude that swap,
  but it isn't built here.

---
*Next: Milestone 12 — Docker, Kubernetes, and deployment.*
