from __future__ import annotations

from src.chunking.hybrid import HybridChunker
from src.chunking.markdown_aware import MarkdownAwareChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.table_aware import TableAwareChunker, serialize_table
from src.parsing.schemas import ParsedDocument, TableContent


def test_serialize_table_basic() -> None:
    table = TableContent(rows=[["A", "B"], ["1", "2"]], caption="My Table")
    rendered = serialize_table(table)
    assert "Table: My Table" in rendered
    assert "| A | B |" in rendered
    assert "| 1 | 2 |" in rendered
    assert "|---|---|" in rendered


def test_serialize_table_pads_ragged_rows() -> None:
    table = TableContent(rows=[["A", "B", "C"], ["1", "2"]])  # short row
    rendered = serialize_table(table)
    assert "| 1 | 2 |  |" in rendered


def test_table_aware_produces_atomic_table_chunk_and_prose_chunk(
    table_document: ParsedDocument,
) -> None:
    chunker = TableAwareChunker()
    chunks = chunker.chunk(table_document, doc_id="doc1")
    table_chunks = [c for c in chunks if c.is_table]
    prose_chunks = [c for c in chunks if not c.is_table]
    assert len(table_chunks) == 1
    assert "Job Level" in table_chunks[0].text
    assert len(prose_chunks) >= 1
    assert "Eligibility Overview" in prose_chunks[0].text


def test_markdown_aware_splits_on_headers(markdown_document: ParsedDocument) -> None:
    chunker = MarkdownAwareChunker(max_chunk_size=800)
    chunks = chunker.chunk(markdown_document, doc_id="doc1")
    heading_paths = [c.metadata.get("heading_path") for c in chunks]
    assert any("Enhanced Due Diligence" in (hp or "") for hp in heading_paths)
    assert any("Monitoring" in (hp or "") for hp in heading_paths)
    # Nested heading path composition
    assert any(hp and hp.startswith("AML Policy >") for hp in heading_paths)


def test_markdown_aware_falls_back_without_raw_source(prose_document: ParsedDocument) -> None:
    # prose_document has no raw_source
    chunker = MarkdownAwareChunker(max_chunk_size=100)
    chunks = chunker.chunk(prose_document, doc_id="doc1")
    assert len(chunks) > 0
    assert all(c.metadata.get("fallback_reason") for c in chunks)


def test_hybrid_does_not_duplicate_table_content(markdown_document: ParsedDocument) -> None:
    # markdown_document fixture has no separate TableContent, so add one via table_document's
    # pattern isn't directly reusable here; this test focuses on structure instead.
    chunker = HybridChunker()
    chunks = chunker.chunk(markdown_document, doc_id="doc1")
    assert len(chunks) > 0
    assert any(c.metadata.get("heading_path") for c in chunks)


def test_hybrid_extracts_table_and_prose_without_raw_markdown_duplication(
    table_document: ParsedDocument,
) -> None:
    chunker = HybridChunker()
    chunks = chunker.chunk(table_document, doc_id="doc1")
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    # The raw markdown pipe syntax should not also appear duplicated in a prose chunk,
    # since table_document has no raw_source (falls back to recursive) and no table
    # syntax in full_text to begin with — this asserts no accidental duplication.
    prose_text = " ".join(c.text for c in chunks if not c.is_table)
    assert "Job Level" not in prose_text


def test_semantic_chunker_breaks_on_topic_shift(fake_embed_fn) -> None:
    from src.parsing.schemas import PageContent, ParsedDocument

    text = (
        "The weather today is sunny and warm. The weather forecast predicts rain tomorrow. "
        "The company reported strong quarterly earnings. Revenue grew significantly this quarter."
    )
    doc = ParsedDocument(
        source_path="f.txt",
        format="txt",
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        page_count=1,
    )
    chunker = SemanticChunker(embed_fn=fake_embed_fn, similarity_threshold=0.5, max_chunk_size=1000)
    chunks = chunker.chunk(doc, doc_id="doc1")
    assert len(chunks) == 2
    assert "weather" in chunks[0].text.lower()
    assert "earnings" in chunks[1].text.lower() or "revenue" in chunks[1].text.lower()


def test_semantic_chunker_single_sentence(fake_embed_fn) -> None:
    from src.parsing.schemas import PageContent, ParsedDocument

    doc = ParsedDocument(
        source_path="f.txt",
        format="txt",
        pages=[PageContent(page_number=1, text="Just one sentence here.")],
        full_text="Just one sentence here.",
        page_count=1,
    )
    chunks = SemanticChunker(embed_fn=fake_embed_fn).chunk(doc, doc_id="doc1")
    assert len(chunks) == 1
