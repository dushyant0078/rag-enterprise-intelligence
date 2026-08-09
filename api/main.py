"""FastAPI Gateway (docs/02_architecture.md's entry point). Every resource
(vector store, embedding model, LLM, pipeline) is built once at startup via
the lifespan context manager and stored on `app.state` -- see api/state.py
and api/dependencies.py. Complete interactive API docs are automatically
available at /docs (Swagger UI) and /redoc, generated from this file's
metadata plus every router's Pydantic schemas -- FastAPI's standard
behavior, not something built separately.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from api.metrics import MetricsRegistry
from api.routers import (
    chat,
    documents,
    evaluate,
    feedback,
    monitoring,
    query,
    search,
    summarize,
    system,
)
from api.state import build_app_state
from src.monitoring.tracing import configure_global_tracing

DESCRIPTION = """
Citation-aware, evaluation-driven Retrieval-Augmented Generation API for
enterprise document collections (banking, insurance, legal, HR, technical
operations). Every answer is grounded against retrieved passages and
rejected in favor of an explicit "insufficient evidence" response rather
than an unsupported claim -- see `docs/07_conversational_rag_pipeline.md`
for how that's enforced, not just requested.

Authenticate with an `X-API-Key` header. See `docs/09_fastapi_backend.md`
for demo credentials and the RBAC model.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Tracing is configured before the pipeline is built below, so
    # ConversationalRAGPipeline's default tracer (trace.get_tracer(), called in its
    # __init__) picks up the real configured provider from the start -- OTel's proxy-tracer
    # design means this ordering isn't strictly required (a tracer obtained before a
    # provider is set still delegates to it once set), but getting it right avoids relying
    # on that subtlety. RAG_TRACE_CONSOLE=true prints every span to stdout for local
    # debugging; off by default so it doesn't spam test/CI output.
    configure_global_tracing(
        console=os.environ.get("RAG_TRACE_CONSOLE", "false").lower() == "true",
        otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    # Data paths are env-var-overridable (RAG_PROCESSED_DIR / RAG_INDEX_DIR /
    # RAG_RAW_UPLOAD_DIR / RAG_METADATA_DIR / RAG_FEEDBACK_DIR), defaulting to the real
    # project paths -- this is both a legitimate production config surface and what lets
    # the test suite point the app at isolated tmp directories instead of polluting real
    # project data.
    app.state.app_state = build_app_state(
        processed_dir=Path(os.environ.get("RAG_PROCESSED_DIR", "data/processed")),
        index_dir=Path(os.environ.get("RAG_INDEX_DIR", "data/embeddings/faiss_index")),
        raw_upload_dir=Path(os.environ.get("RAG_RAW_UPLOAD_DIR", "data/raw/api_uploads")),
        metadata_dir=Path(os.environ.get("RAG_METADATA_DIR", "data/metadata")),
        feedback_dir=Path(os.environ.get("RAG_FEEDBACK_DIR", "data/feedback")),
    )
    app.state.metrics_registry = MetricsRegistry()
    yield
    # No explicit teardown needed: FAISS/session state is either already persisted to disk
    # (FaissVectorStore.persist() is called inside build_index) or in-memory-only by design.


app = FastAPI(
    title="RAG Enterprise Intelligence API",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)

# Real OTel HTTP-level auto-instrumentation (Milestone 11) -- every request gets a span
# automatically, on top of ConversationalRAGPipeline's own per-stage spans. Safe to call
# once at module scope even before `configure_global_tracing()` runs in lifespan: OTel's
# proxy-tracer design means instrumentation created against "whatever provider is active"
# correctly picks up the real provider once lifespan configures one.
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    # Use the matched route template ("/chat", not the literal request path) so metrics
    # aggregate correctly regardless of query params/path variables.
    route = request.scope.get("route")
    path = route.path if route is not None else request.url.path
    request.app.state.metrics_registry.record(request.method, path, response.status_code, duration)
    return response


app.include_router(system.router)
app.include_router(chat.router)
app.include_router(query.router)
app.include_router(search.router)
app.include_router(summarize.router)
app.include_router(documents.router)
app.include_router(evaluate.router)
app.include_router(feedback.router)
app.include_router(monitoring.router)
