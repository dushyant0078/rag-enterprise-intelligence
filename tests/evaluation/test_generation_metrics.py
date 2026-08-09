from __future__ import annotations

from src.citations.schemas import Citation
from src.evaluation.generation_metrics import (
    answer_correctness_lexical,
    citation_accuracy,
    faithfulness_lexical,
    hallucination_rate,
)
from src.retrieval.schemas import RetrievedChunk


def _citation(chunk_id: str | None, doc_id: str = "d1") -> Citation:
    return Citation(doc_id=doc_id, chunk_id=chunk_id, confidence=0.9 if chunk_id else 0.0)


def test_citation_accuracy_all_resolved() -> None:
    assert citation_accuracy([_citation("c1"), _citation("c2")]) == 1.0


def test_citation_accuracy_partial() -> None:
    assert citation_accuracy([_citation("c1"), _citation(None)]) == 0.5


def test_citation_accuracy_empty_list() -> None:
    assert citation_accuracy([]) == 0.0


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", doc_id="d1", text=text, score=0.9)


def test_hallucination_rate_fully_supported() -> None:
    chunks = [_chunk("Wire transfers over $10,000 require enhanced due diligence review.")]
    answer = "Wire transfers over $10,000 require enhanced due diligence review."
    assert hallucination_rate(answer, chunks) == 0.0


def test_hallucination_rate_fully_unsupported() -> None:
    chunks = [_chunk("Wire transfers over $10,000 require enhanced due diligence review.")]
    answer = "The capital of France is Paris and it has a large population."
    assert hallucination_rate(answer, chunks) == 1.0


def test_hallucination_rate_mixed() -> None:
    chunks = [_chunk("Wire transfers over $10,000 require enhanced due diligence review.")]
    answer = (
        "Wire transfers over $10,000 require enhanced due diligence review. "
        "The capital of France is Paris."
    )
    assert hallucination_rate(answer, chunks) == 0.5


def test_hallucination_rate_no_sentences_returns_zero() -> None:
    assert hallucination_rate("", [_chunk("text")]) == 0.0


def test_faithfulness_is_complement_of_hallucination_rate() -> None:
    chunks = [_chunk("Wire transfers require enhanced due diligence.")]
    answer = "Wire transfers require enhanced due diligence. Unrelated fact about France."
    assert (
        abs(faithfulness_lexical(answer, chunks) + hallucination_rate(answer, chunks) - 1.0) < 1e-9
    )


def test_answer_correctness_identical_is_one() -> None:
    assert answer_correctness_lexical("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_answer_correctness_disjoint_is_zero() -> None:
    assert (
        answer_correctness_lexical("completely different words", "nothing overlapping here") == 0.0
    )


def test_answer_correctness_partial_overlap_between_zero_and_one() -> None:
    score = answer_correctness_lexical("the cat sat on the mat today", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_answer_correctness_empty_generated_is_zero() -> None:
    assert answer_correctness_lexical("", "some expected answer") == 0.0
