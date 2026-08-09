from __future__ import annotations

from src.citations.response_validator import ResponseValidator
from src.prompting.guardrails import INSUFFICIENT_EVIDENCE_PHRASE
from src.retrieval.schemas import RetrievedChunk


def _chunk(chunk_id: str, doc_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=doc_id, text="Some source text.", score=score)


def test_grounded_answer_with_high_confidence_citation_passes() -> None:
    chunks = [_chunk("c1", "doc1", 0.9)]
    result = ResponseValidator(min_confidence=0.7).validate(
        "The answer is X [Source: doc1].", chunks
    )
    assert result.is_grounded
    assert result.answer_text == "The answer is X [Source: doc1]."
    assert result.confidence_score == 0.9


def test_answer_with_no_citations_is_not_grounded() -> None:
    chunks = [_chunk("c1", "doc1", 0.9)]
    result = ResponseValidator().validate("A confident-sounding answer with no citation.", chunks)
    assert not result.is_grounded
    assert result.answer_text == INSUFFICIENT_EVIDENCE_PHRASE
    assert result.confidence_score == 0.0


def test_fabricated_citation_invalidates_the_whole_answer() -> None:
    chunks = [_chunk("c1", "doc1", 0.9)]
    result = ResponseValidator().validate(
        "Fact [Source: doc1]. Other fact [Source: fake-doc].", chunks
    )
    assert not result.is_grounded
    assert result.answer_text == INSUFFICIENT_EVIDENCE_PHRASE
    assert len(result.fabricated_citations) == 1
    assert result.fabricated_citations[0].doc_id == "fake-doc"


def test_low_confidence_citation_below_threshold_fails() -> None:
    chunks = [_chunk("c1", "doc1", 0.3)]  # below default 0.7 threshold
    result = ResponseValidator().validate("Fact [Source: doc1].", chunks)
    assert not result.is_grounded
    assert result.answer_text == INSUFFICIENT_EVIDENCE_PHRASE


def test_custom_min_confidence_threshold() -> None:
    chunks = [_chunk("c1", "doc1", 0.5)]
    lenient = ResponseValidator(min_confidence=0.3).validate("Fact [Source: doc1].", chunks)
    strict = ResponseValidator(min_confidence=0.9).validate("Fact [Source: doc1].", chunks)
    assert lenient.is_grounded
    assert not strict.is_grounded


def test_original_answer_text_always_preserved() -> None:
    chunks = [_chunk("c1", "doc1", 0.9)]
    original = "A confident-sounding answer with no citation."
    result = ResponseValidator().validate(original, chunks)
    assert result.original_answer_text == original
    assert result.answer_text != original  # replaced with fallback


def test_retrieved_passages_carried_through() -> None:
    chunks = [_chunk("c1", "doc1", 0.9), _chunk("c2", "doc2", 0.5)]
    result = ResponseValidator().validate("Fact [Source: doc1].", chunks)
    assert result.retrieved_passages == chunks


def test_average_confidence_across_multiple_resolved_citations() -> None:
    chunks = [_chunk("c1", "doc1", 1.0), _chunk("c2", "doc2", 0.6)]
    result = ResponseValidator(min_confidence=0.5).validate(
        "Fact one [Source: doc1]. Fact two [Source: doc2].", chunks
    )
    assert abs(result.confidence_score - 0.8) < 1e-9
    assert result.is_grounded
