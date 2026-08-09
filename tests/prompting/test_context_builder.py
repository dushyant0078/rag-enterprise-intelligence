from __future__ import annotations

from src.prompting.context_builder import ContextBuilder
from src.retrieval.schemas import RetrievedChunk


def _chunk(
    chunk_id: str, text: str, score: float = 0.5, page_numbers: list[int] | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        score=score,
        page_numbers=page_numbers or [],
    )


def test_empty_chunks_returns_empty_string() -> None:
    assert ContextBuilder().build([]) == ""


def test_includes_citation_tag_with_page_number() -> None:
    chunks = [_chunk("a", "Some chunk text.", page_numbers=[3])]
    context = ContextBuilder().build(chunks)
    assert "[Source: doc-a, page 3]" in context
    assert "Some chunk text." in context


def test_citation_tag_without_page_number() -> None:
    chunks = [_chunk("a", "Some chunk text.")]
    context = ContextBuilder().build(chunks)
    assert "[Source: doc-a]" in context


def test_multiple_chunks_separated_by_delimiter() -> None:
    chunks = [_chunk("a", "First chunk."), _chunk("b", "Second chunk.")]
    context = ContextBuilder().build(chunks)
    assert "First chunk." in context
    assert "Second chunk." in context
    assert "---" in context


def test_respects_token_budget() -> None:
    # Each chunk is roughly 100 tokens (400 chars / 4); a budget of 150 tokens should
    # include the first chunk but not a second one that would exceed it.
    long_text = "word " * 100
    chunks = [_chunk("a", long_text), _chunk("b", long_text)]
    builder = ContextBuilder(max_context_tokens=150)
    context = builder.build(chunks)
    assert "doc-a" in context
    assert "doc-b" not in context


def test_always_includes_at_least_one_chunk_even_if_over_budget() -> None:
    huge_text = "word " * 10000
    builder = ContextBuilder(max_context_tokens=10)
    context = builder.build([_chunk("a", huge_text)])
    assert "doc-a" in context


def test_included_chunk_count_matches_build_output() -> None:
    long_text = "word " * 100
    chunks = [_chunk("a", long_text), _chunk("b", long_text), _chunk("c", long_text)]
    builder = ContextBuilder(max_context_tokens=150)
    count = builder.included_chunk_count(chunks)
    context = builder.build(chunks)
    assert count == 1
    assert context.count("[Source:") == count


def test_included_chunk_count_empty_list() -> None:
    assert ContextBuilder().included_chunk_count([]) == 0
