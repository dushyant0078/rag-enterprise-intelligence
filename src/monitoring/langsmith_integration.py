"""LangSmith tracing integration.

**Untestable in this sandbox for the usual network reason** (confirmed
directly: `api.smith.langchain.com` returns `x-deny-reason: host_not_allowed`
at the egress proxy, the same synthetic-block signature as OpenAI/Google/
Weaviate/Pinecone/Cohere elsewhere in this project -- not a real API
response). `langsmith` imports cleanly (verified), so the functions below
are written and import-checked against the real installed SDK; what's not
verified is an actual traced run reaching LangSmith's backend.

LangSmith and this project's own OpenTelemetry tracing (tracing.py) are
complementary, not redundant: OTel spans are self-hosted (export to any
OTLP collector, or nowhere); LangSmith is a hosted, LLM-specific observability
product with its own UI for prompt/response inspection, evaluation runs,
and dataset curation. A deployment can run either or both.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def is_configured() -> bool:
    """True if LANGSMITH_API_KEY is set. Callers should check this before
    relying on tracing actually reaching LangSmith -- `traceable` degrades
    to a no-op decorator without it (real langsmith SDK behavior), so
    nothing crashes in an unconfigured environment like this sandbox.
    """
    import os

    return bool(os.environ.get("LANGSMITH_API_KEY"))


def traceable_rag_call(name: str) -> Callable[[F], F]:
    """Wraps `langsmith.traceable` for tagging a pipeline stage as a traced
    run. Written directly from langsmith's documented decorator API and
    import-verified against the installed SDK; not exercised against a real
    LangSmith project here (see module docstring).
    """
    from langsmith import traceable

    def decorator(func: F) -> F:
        return traceable(name=name, run_type="chain")(func)  # type: ignore[return-value]

    return decorator


def log_feedback(run_id: str, key: str, score: float, comment: str | None = None) -> None:
    """Push feedback (e.g. from src/monitoring/feedback.py's thumbs up/down)
    to a LangSmith run so it shows up alongside that run's trace in
    LangSmith's UI. Requires LANGSMITH_API_KEY; raises if not configured
    rather than silently doing nothing, since a caller explicitly asking to
    log feedback should know if it didn't happen.
    """
    if not is_configured():
        raise RuntimeError("LANGSMITH_API_KEY is not set -- cannot log feedback to LangSmith")

    from langsmith import Client

    client = Client()
    client.create_feedback(run_id=run_id, key=key, score=score, comment=comment)
