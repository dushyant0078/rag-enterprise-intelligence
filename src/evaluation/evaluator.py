"""RAGEvaluator: the orchestrator that formalizes what Milestones 4-7's
comparison scripts each did ad hoc (build an index, run hand-written
queries, print a hit rate) into one reusable framework -- runs a labeled
BenchmarkDataset through a real ConversationalRAGPipeline and computes every
metric in src/evaluation/ for each query, then aggregates.
"""

from __future__ import annotations

import time

from src.evaluation.benchmark_dataset import BenchmarkDataset
from src.evaluation.generation_metrics import (
    answer_correctness_lexical,
    citation_accuracy,
    faithfulness_lexical,
    hallucination_rate,
)
from src.evaluation.latency import summarize_latencies
from src.evaluation.retrieval_metrics import (
    context_precision,
    context_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.schemas import (
    AggregateReport,
    BenchmarkQuery,
    GenerationMetrics,
    LatencyMetrics,
    QueryEvaluationResult,
    RetrievalMetrics,
)
from src.memory.rag_pipeline import ConversationalRAGPipeline


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class RAGEvaluator:
    def __init__(self, pipeline: ConversationalRAGPipeline, k: int = 5) -> None:
        self.pipeline = pipeline
        self.k = k

    def evaluate_query(
        self, benchmark_query: BenchmarkQuery, user_id: str = "evaluator"
    ) -> QueryEvaluationResult:
        # A fresh session per query: benchmark queries are evaluated independently, not as
        # one running conversation -- reusing a session would let query rewriting or
        # accumulated history leak between unrelated benchmark items.
        session_id = self.pipeline.start_session(user_id=user_id)

        start = time.monotonic()
        result = self.pipeline.ask(session_id, benchmark_query.query)
        total_ms = (time.monotonic() - start) * 1000

        retrieved_doc_ids = [c.doc_id for c in result.retrieved_passages]
        relevant_ids = set(benchmark_query.relevant_doc_ids)

        retrieval_metrics = RetrievalMetrics(
            precision_at_k=precision_at_k(retrieved_doc_ids, relevant_ids, self.k),
            recall_at_k=recall_at_k(retrieved_doc_ids, relevant_ids, self.k),
            mrr=mrr(retrieved_doc_ids, relevant_ids),
            ndcg_at_k=ndcg_at_k(retrieved_doc_ids, relevant_ids, self.k),
            context_precision=context_precision(retrieved_doc_ids, relevant_ids),
            context_recall=context_recall(retrieved_doc_ids, relevant_ids),
            k=self.k,
        )

        generation_metrics = GenerationMetrics(
            citation_accuracy=citation_accuracy(result.citations),
            hallucination_rate=hallucination_rate(result.answer_text, result.retrieved_passages),
            faithfulness=faithfulness_lexical(result.answer_text, result.retrieved_passages),
            answer_correctness=answer_correctness_lexical(
                result.answer_text, benchmark_query.expected_answer
            ),
            is_grounded=result.is_grounded,
        )

        # Only overall (retrieval + generation combined) latency is measured end-to-end here;
        # a retrieval-only breakdown would need the pipeline to report per-stage timing, which
        # it doesn't yet (a natural extension for Milestone 11's monitoring instrumentation).
        generation_ms = result.metadata.get("latency_ms") or 0.0
        latency_metrics = LatencyMetrics(
            retrieval_ms=max(0.0, total_ms - generation_ms),
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

        return QueryEvaluationResult(
            query_id=benchmark_query.query_id,
            query=benchmark_query.query,
            answer_text=result.answer_text,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_metrics=latency_metrics,
        )

    def evaluate(self, dataset: BenchmarkDataset, user_id: str = "evaluator") -> AggregateReport:
        results = [self.evaluate_query(q, user_id=user_id) for q in dataset]

        latencies = [r.latency_metrics.total_ms for r in results]
        lat_summary = summarize_latencies(latencies)

        return AggregateReport(
            benchmark_size=len(results),
            k=self.k,
            mean_precision_at_k=_mean([r.retrieval_metrics.precision_at_k for r in results]),
            mean_recall_at_k=_mean([r.retrieval_metrics.recall_at_k for r in results]),
            mean_mrr=_mean([r.retrieval_metrics.mrr for r in results]),
            mean_ndcg_at_k=_mean([r.retrieval_metrics.ndcg_at_k for r in results]),
            mean_context_precision=_mean([r.retrieval_metrics.context_precision for r in results]),
            mean_context_recall=_mean([r.retrieval_metrics.context_recall for r in results]),
            mean_citation_accuracy=_mean([r.generation_metrics.citation_accuracy for r in results]),
            mean_hallucination_rate=_mean(
                [r.generation_metrics.hallucination_rate for r in results]
            ),
            mean_faithfulness=_mean([r.generation_metrics.faithfulness for r in results]),
            mean_answer_correctness=_mean(
                [r.generation_metrics.answer_correctness for r in results]
            ),
            grounded_rate=_mean(
                [1.0 if r.generation_metrics.is_grounded else 0.0 for r in results]
            ),
            latency_p50_ms=lat_summary["p50"],
            latency_p95_ms=lat_summary["p95"],
            latency_mean_ms=lat_summary["mean"],
            per_query=results,
        )
