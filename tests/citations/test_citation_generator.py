from __future__ import annotations

from src.citations.citation_generator import CitationGenerator
from src.retrieval.schemas import RetrievedChunk


def _chunk(
    chunk_id: str, doc_id: str, text: str, score: float, page_numbers=None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id=doc_id, text=text, score=score, page_numbers=page_numbers or []
    )


def test_parses_single_citation_with_page() -> None:
    chunks = [_chunk("c1", "doc1", "Some text.", 0.9, page_numbers=[3])]
    citations = CitationGenerator().generate("The answer is X [Source: doc1, page 3].", chunks)
    assert len(citations) == 1
    assert citations[0].doc_id == "doc1"
    assert citations[0].page_number == 3
    assert citations[0].chunk_id == "c1"
    assert citations[0].confidence == 0.9
    assert not citations[0].is_fabricated


def test_parses_citation_without_page() -> None:
    chunks = [_chunk("c1", "doc1", "Some text.", 0.7)]
    citations = CitationGenerator().generate("The answer is X [Source: doc1].", chunks)
    assert citations[0].page_number is None
    assert citations[0].chunk_id == "c1"


def test_parses_multiple_citations() -> None:
    chunks = [_chunk("c1", "doc1", "Text one.", 0.9), _chunk("c2", "doc2", "Text two.", 0.8)]
    citations = CitationGenerator().generate(
        "Fact one [Source: doc1]. Fact two [Source: doc2].", chunks
    )
    assert len(citations) == 2
    assert {c.doc_id for c in citations} == {"doc1", "doc2"}


def test_fabricated_citation_not_in_chunks() -> None:
    chunks = [_chunk("c1", "doc1", "Text.", 0.9)]
    citations = CitationGenerator().generate("Fact [Source: nonexistent-doc].", chunks)
    assert citations[0].is_fabricated
    assert citations[0].chunk_id is None
    assert citations[0].confidence == 0.0


def test_no_citations_in_answer_returns_empty_list() -> None:
    chunks = [_chunk("c1", "doc1", "Text.", 0.9)]
    citations = CitationGenerator().generate("An answer with no citation tags at all.", chunks)
    assert citations == []


def test_page_mismatch_falls_back_to_any_chunk_from_same_doc() -> None:
    chunks = [_chunk("c1", "doc1", "Text.", 0.9, page_numbers=[5])]
    citations = CitationGenerator().generate("Fact [Source: doc1, page 99].", chunks)
    # page 99 doesn't exist for doc1, but doc1 itself is real -- falls back rather than
    # treating a page-number typo as full fabrication.
    assert not citations[0].is_fabricated
    assert citations[0].chunk_id == "c1"


def test_prefers_exact_page_match_when_multiple_chunks_share_doc_id() -> None:
    chunks = [
        _chunk("c1", "doc1", "Page one text.", 0.5, page_numbers=[1]),
        _chunk("c2", "doc1", "Page two text.", 0.9, page_numbers=[2]),
    ]
    citations = CitationGenerator().generate("Fact [Source: doc1, page 2].", chunks)
    assert citations[0].chunk_id == "c2"
    assert citations[0].confidence == 0.9


def test_excerpt_is_truncated_from_matched_chunk() -> None:
    long_text = "word " * 100
    chunks = [_chunk("c1", "doc1", long_text, 0.9)]
    citations = CitationGenerator().generate("Fact [Source: doc1].", chunks)
    assert len(citations[0].excerpt) <= 200
