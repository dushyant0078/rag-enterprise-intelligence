from __future__ import annotations

import pytest

from src.citations.schemas import ValidatedAnswer
from src.evaluation.benchmark_dataset import default_sample_corpus_benchmark
from src.evaluation.deepeval_integration import build_metrics, build_test_case
from src.retrieval.schemas import RetrievedChunk


def _sample_result() -> ValidatedAnswer:
    return ValidatedAnswer(
        answer_text="International wire transfers over $3,000 require enhanced due diligence.",
        original_answer_text="...",
        confidence_score=0.9,
        is_grounded=True,
        retrieved_passages=[
            RetrievedChunk(chunk_id="c1", doc_id="banking-doc", text="Some source text.", score=0.9)
        ],
    )


def test_build_test_case_constructs_real_llmtestcase() -> None:
    query = default_sample_corpus_benchmark()[0]
    test_case = build_test_case(query, _sample_result())
    assert test_case.input == query.query
    assert test_case.actual_output == _sample_result().answer_text
    assert test_case.expected_output == query.expected_answer
    assert test_case.retrieval_context == ["Some source text."]


def test_build_metrics_raises_without_configured_judge() -> None:
    """Confirms the exact, real failure point: metric *construction* itself
    needs an LLM judge, not just measure() -- verified directly against the
    installed deepeval package, not assumed from its docs.
    """
    with pytest.raises(Exception, match="(?i)api.?key|not configured"):
        build_metrics()
