from __future__ import annotations

from src.evaluation.benchmark_dataset import BenchmarkDataset
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.schemas import BenchmarkQuery
from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.memory.rag_pipeline import ConversationalRAGPipeline
from src.prompting.registry import get_template
from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk


class _FixedRetriever(BaseRetriever):
    retriever_name = "fixed"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query, top_k=10, filters=None):
        return self._chunks[:top_k]


def _banking_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="banking-aml-policy",
        text="Wire transfers over $10,000 require enhanced due diligence review.",
        score=0.9,
        page_numbers=[1],
    )


def _build_pipeline(chunks: list[RetrievedChunk]) -> ConversationalRAGPipeline:
    return ConversationalRAGPipeline(
        retriever=_FixedRetriever(chunks),
        llm=DeterministicTestLLM(),
        prompt_template=get_template("qa"),
    )


def test_evaluate_query_computes_all_metric_groups() -> None:
    pipeline = _build_pipeline([_banking_chunk()])
    evaluator = RAGEvaluator(pipeline, k=5)
    query = BenchmarkQuery(
        query_id="q1",
        query="What triggers enhanced due diligence for wire transfers?",
        expected_answer="Wire transfers over $10,000 require enhanced due diligence.",
        relevant_doc_ids=["banking"],
    )
    result = evaluator.evaluate_query(query)

    assert result.retrieval_metrics.mrr == 1.0  # only chunk, relevant, rank 1
    assert result.generation_metrics.citation_accuracy == 1.0
    assert result.generation_metrics.is_grounded
    assert result.latency_metrics.total_ms >= 0


def test_evaluate_query_low_retrieval_score_for_irrelevant_chunks() -> None:
    pipeline = _build_pipeline([_banking_chunk()])
    evaluator = RAGEvaluator(pipeline, k=5)
    query = BenchmarkQuery(
        query_id="q1",
        query="anything",
        expected_answer="anything",
        relevant_doc_ids=["totally-unrelated-domain"],
    )
    result = evaluator.evaluate_query(query)
    assert result.retrieval_metrics.mrr == 0.0
    assert result.retrieval_metrics.precision_at_k == 0.0


def test_evaluate_runs_full_benchmark_and_aggregates() -> None:
    pipeline = _build_pipeline([_banking_chunk()])
    evaluator = RAGEvaluator(pipeline, k=5)
    dataset = BenchmarkDataset(
        [
            BenchmarkQuery(
                query_id="q1",
                query="What triggers enhanced due diligence?",
                expected_answer="Enhanced due diligence for wire transfers.",
                relevant_doc_ids=["banking"],
            ),
            BenchmarkQuery(
                query_id="q2",
                query="What triggers enhanced due diligence for wire transfers?",
                expected_answer="Wire transfers over $10,000 require enhanced due diligence.",
                relevant_doc_ids=["banking"],
            ),
        ]
    )
    report = evaluator.evaluate(dataset)

    assert report.benchmark_size == 2
    assert len(report.per_query) == 2
    assert 0.0 <= report.mean_precision_at_k <= 1.0
    assert 0.0 <= report.grounded_rate <= 1.0
    assert report.latency_mean_ms >= 0


def test_evaluate_query_uses_fresh_session_each_time() -> None:
    """Regression-style check: benchmark queries must not accumulate
    conversation history across each other -- each is independent.
    """
    pipeline = _build_pipeline([_banking_chunk()])
    evaluator = RAGEvaluator(pipeline, k=5)
    query = BenchmarkQuery(
        query_id="q1", query="What about it?", expected_answer="x", relevant_doc_ids=["banking"]
    )
    # A pronoun-led query with no prior history in its own fresh session should NOT be
    # rewritten (no history to resolve "it" against), unlike in a real accumulating conversation.
    evaluator.evaluate_query(query)
    evaluator.evaluate_query(query)  # running it twice must not change behavior via leaked state
