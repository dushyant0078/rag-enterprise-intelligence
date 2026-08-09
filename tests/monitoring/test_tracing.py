from __future__ import annotations

from src.monitoring.tracing import InMemorySpanExporter, build_isolated_tracer_provider


def test_isolated_provider_captures_real_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = build_isolated_tracer_provider(exporter)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("my-span") as span:
        span.set_attribute("foo", "bar")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my-span"
    assert spans[0].attributes["foo"] == "bar"


def test_isolated_providers_do_not_interfere() -> None:
    exporter_a = InMemorySpanExporter()
    exporter_b = InMemorySpanExporter()
    tracer_a = build_isolated_tracer_provider(exporter_a).get_tracer("a")
    build_isolated_tracer_provider(exporter_b).get_tracer("b")  # constructed, deliberately unused

    with tracer_a.start_as_current_span("span-a"):
        pass

    assert len(exporter_a.get_finished_spans()) == 1
    assert len(exporter_b.get_finished_spans()) == 0


def test_pipeline_stages_produce_real_spans_with_attributes() -> None:
    """End-to-end: a real ConversationalRAGPipeline.ask() call, with an
    injected isolated tracer, produces one real span per documented stage.
    """
    from pathlib import Path

    from src.memory.pipeline_factory import build_default_pipeline

    exporter = InMemorySpanExporter()
    tracer = build_isolated_tracer_provider(exporter).get_tracer("test")

    # A minimal isolated corpus, not the real project data dir.
    import json

    from src.parsing.schemas import PageContent, ParsedDocument

    processed_dir = Path("/tmp/tracing_test_processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    doc = ParsedDocument(
        source_path="t.txt",
        format="txt",
        pages=[PageContent(page_number=1, text="Enhanced due diligence applies over $10,000.")],
        full_text="Enhanced due diligence applies over $10,000.",
        page_count=1,
    )
    domain_dir = processed_dir / "banking"
    domain_dir.mkdir(exist_ok=True)
    (domain_dir / "banking__doc1.json").write_text(json.dumps(json.loads(doc.model_dump_json())))

    pipeline = build_default_pipeline(
        processed_dir=processed_dir, index_dir=Path("/tmp/tracing_test_index")
    )
    pipeline._tracer = tracer  # inject after construction -- simplest way to reuse the factory

    session_id = pipeline.start_session(user_id="test")
    result = pipeline.ask(session_id, "What is the enhanced due diligence threshold?")

    span_names = {s.name for s in exporter.get_finished_spans()}
    assert span_names == {
        "conversation_manager.prepare_turn",
        "retriever.retrieve",
        "reranker.rerank",
        "context_builder.build",
        "llm.generate",
        "response_validator.validate",
    }
    assert "stage_latencies_ms" in result.metadata
    assert set(result.metadata["stage_latencies_ms"].keys()) >= {
        "retrieval_ms",
        "generation_ms",
        "total_ms",
    }
