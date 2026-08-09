from __future__ import annotations

from src.chunking.recursive import RecursiveChunker
from src.parsing.schemas import PageContent, ParsedDocument
from src.retrieval.parent_child import ParentChildRetriever, build_parent_child_chunks
from tests.retrieval.conftest import FakeRetriever, make_chunk


def _make_doc() -> ParsedDocument:
    text = (
        "Section 1: Remote Work Eligibility. Individual contributors may work remotely up to "
        "five days per week with manager approval. Team leads may work remotely up to four "
        "days.\n\n"
        "Section 2: Parental Leave Interaction. Employees returning from parental leave retain "
        "their prior remote-days eligibility for ninety days before standard rules resume."
    )
    return ParsedDocument(
        source_path="f.txt",
        format="txt",
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        page_count=1,
    )


def test_build_parent_child_chunks_every_child_has_a_parent() -> None:
    doc = _make_doc()
    parents, children = build_parent_child_chunks(
        doc,
        doc_id="doc1",
        parent_chunker=RecursiveChunker(chunk_size=200, chunk_overlap=0),
        child_chunker=RecursiveChunker(chunk_size=60, chunk_overlap=0),
    )
    assert len(parents) >= 1
    assert len(children) >= len(parents)

    parent_ids = {p.chunk_id for p in parents}
    for child in children:
        assert child.metadata["parent_chunk_id"] in parent_ids


def test_build_parent_child_children_are_smaller_than_parents() -> None:
    doc = _make_doc()
    parents, children = build_parent_child_chunks(
        doc,
        doc_id="doc1",
        parent_chunker=RecursiveChunker(chunk_size=300, chunk_overlap=0),
        child_chunker=RecursiveChunker(chunk_size=50, chunk_overlap=0),
    )
    max_parent_len = max(len(p.text) for p in parents)
    max_child_len = max(len(c.text) for c in children)
    assert max_child_len <= max_parent_len


def test_parent_child_retriever_returns_parent_text() -> None:
    doc = _make_doc()
    parents, children = build_parent_child_chunks(doc, doc_id="doc1")
    parent_lookup = {p.chunk_id: p for p in parents}

    # Simulate the top-matching child being from the first parent.
    matched_child = children[0]
    child_retriever = FakeRetriever(
        [
            make_chunk(
                matched_child.chunk_id,
                0.9,
                metadata={"parent_chunk_id": matched_child.metadata["parent_chunk_id"]},
            )
        ]
    )
    retriever = ParentChildRetriever(child_retriever, parent_lookup)

    results = retriever.retrieve("remote work eligibility", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == matched_child.metadata["parent_chunk_id"]
    assert results[0].text == parent_lookup[matched_child.metadata["parent_chunk_id"]].text
    assert results[0].source_retriever == "parent_child"


def test_parent_child_retriever_dedupes_multiple_children_same_parent() -> None:
    doc = _make_doc()
    parents, children = build_parent_child_chunks(doc, doc_id="doc1")
    parent_lookup = {p.chunk_id: p for p in parents}
    same_parent_children = [
        c for c in children if c.metadata["parent_chunk_id"] == parents[0].chunk_id
    ]
    assert len(same_parent_children) >= 1

    child_retriever = FakeRetriever(
        [
            make_chunk(
                c.chunk_id, 0.9 - i * 0.01, metadata={"parent_chunk_id": parents[0].chunk_id}
            )
            for i, c in enumerate(same_parent_children)
        ]
    )
    retriever = ParentChildRetriever(child_retriever, parent_lookup)
    results = retriever.retrieve("query", top_k=5)
    assert len(results) == 1  # deduped to one parent despite multiple matching children


def test_parent_child_retriever_skips_missing_parent() -> None:
    child_retriever = FakeRetriever(
        [make_chunk("orphan_child", 0.9, metadata={"parent_chunk_id": "does_not_exist"})]
    )
    retriever = ParentChildRetriever(child_retriever, parent_lookup={})
    assert retriever.retrieve("query", top_k=5) == []
