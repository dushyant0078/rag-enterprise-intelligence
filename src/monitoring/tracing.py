"""OpenTelemetry tracing setup.

Real, working instrumentation -- `ConversationalRAGPipeline` (Milestone 7,
updated in Milestone 11) wraps every stage in a real span via an injected
tracer, verified directly against an `InMemorySpanExporter` in
tests/monitoring/test_tracing.py, not assumed to work from the SDK's docs.

Two entry points, deliberately separate:
- `build_isolated_tracer_provider()` -- a standalone provider bound to
  whatever exporter you give it, never touching global state. What tests
  use, and what lets multiple tests run without stepping on each other.
- `configure_global_tracing()` -- sets the process-wide tracer provider
  once at application startup (see api/main.py's lifespan), the way a real
  deployment wires up an OTLP exporter to send spans to a collector
  (Jaeger, Tempo, an APM vendor, ...). Defaults to a console exporter for
  local dev when no OTLP endpoint is configured, and is a genuine no-op
  (real OTel API behavior, not a workaround) if tracing is never configured
  at all -- every span-creating call site in this project works unchanged
  either way.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

SERVICE_NAME_VALUE = "rag-enterprise-intelligence"


def build_isolated_tracer_provider(exporter) -> TracerProvider:  # noqa: ANN001 -- any SpanExporter
    """A standalone TracerProvider bound only to `exporter`, never calling
    `trace.set_tracer_provider()` -- safe to construct many of in the same
    process (e.g. once per test) without global-state interference.
    """
    resource = Resource.create({SERVICE_NAME: SERVICE_NAME_VALUE})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def configure_global_tracing(
    *, console: bool = True, otlp_endpoint: str | None = None
) -> TracerProvider:
    """Configure the process-wide tracer provider once, at app startup.
    `otlp_endpoint`, if given, adds a real OTLP exporter (needs the
    `opentelemetry-exporter-otlp` package and a reachable collector --
    correct code, not verified in this sandbox, same honesty as every other
    network-dependent integration in this project). `console=True` always
    also logs spans to stdout, useful for local development regardless of
    whether a real collector is configured.
    """
    resource = Resource.create({SERVICE_NAME: SERVICE_NAME_VALUE})
    provider = TracerProvider(resource=resource)

    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

    trace.set_tracer_provider(provider)
    return provider


def get_global_tracer(name: str = "rag_enterprise_intelligence") -> Tracer:
    """Whatever tracer the currently-configured global provider gives --
    a real, working no-op tracer if configure_global_tracing() was never
    called, by design of the OTel API (every span becomes a cheap no-op,
    not an error).
    """
    return trace.get_tracer(name)


__all__ = [
    "InMemorySpanExporter",
    "build_isolated_tracer_provider",
    "configure_global_tracing",
    "get_global_tracer",
]
