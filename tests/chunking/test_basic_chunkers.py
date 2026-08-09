from __future__ import annotations

import pytest

from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.sentence_aware import SentenceAwareChunker
from src.parsing.schemas import ParsedDocument


def test_fixed_size_respects_chunk_size(prose_document: ParsedDocument) -> None:
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(prose_document, doc_id="doc1")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100
        assert c.strategy == "fixed_size"
        assert c.doc_id == "doc1"


def test_fixed_size_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)


def test_fixed_size_empty_document_returns_no_chunks(empty_document: ParsedDocument) -> None:
    assert FixedSizeChunker().chunk(empty_document, doc_id="doc1") == []


def test_recursive_chunker_respects_chunk_size(prose_document: ParsedDocument) -> None:
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(prose_document, doc_id="doc1")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 150 + 20  # merged overlap may push slightly over the raw cap


def test_recursive_chunker_prefers_paragraph_boundaries(prose_document: ParsedDocument) -> None:
    # With a size cap larger than any single paragraph, each paragraph should
    # land in its own chunk rather than being sliced mid-sentence.
    chunker = RecursiveChunker(
        chunk_size=250, chunk_overlap=0, separators=("\n\n", "\n", ". ", " ")
    )
    chunks = chunker.chunk(prose_document, doc_id="doc1")
    for c in chunks:
        assert not c.text.strip().endswith(("employe", "arrangeme"))  # no obvious mid-word cut


def test_recursive_chunker_deterministic_ids(prose_document: ParsedDocument) -> None:
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(prose_document, doc_id="doc1")
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"doc1__recursive__{i:04d}"


def test_sentence_aware_never_splits_mid_sentence(prose_document: ParsedDocument) -> None:
    chunker = SentenceAwareChunker(max_chunk_size=120)
    chunks = chunker.chunk(prose_document, doc_id="doc1")
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?"))


def test_sentence_aware_empty_document(empty_document: ParsedDocument) -> None:
    assert SentenceAwareChunker().chunk(empty_document, doc_id="doc1") == []
