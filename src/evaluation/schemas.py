"""Data model for src/evaluation/: a labeled benchmark query, and the
per-query / aggregate results the evaluator produces.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    """One labeled example: a query plus the ground truth needed to score
    retrieval (relevant_doc_ids) and generation (expected_answer) against.
    """

    query_id: str
    query: str
    expected_answer: str
    relevant_doc_ids: list[str]  # doc_id prefixes/values considered a correct retrieval
    domain: str | None = None


class RetrievalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    context_precision: float
    context_recall: float
    k: int


class GenerationMetrics(BaseModel):
    citation_accuracy: float  # fraction of citations that resolved to a real chunk
    hallucination_rate: float  # fraction of answer sentences with no lexical support in context
    faithfulness: float  # 1 - hallucination_rate (offline proxy -- see generation_metrics.py)
    answer_correctness: float  # lexical F1 vs. expected_answer (offline proxy)
    is_grounded: bool


class LatencyMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryEvaluationResult(BaseModel):
    query_id: str
    query: str
    answer_text: str
    retrieved_doc_ids: list[str]
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    latency_metrics: LatencyMetrics


class AggregateReport(BaseModel):
    benchmark_size: int
    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_context_precision: float
    mean_context_recall: float
    mean_citation_accuracy: float
    mean_hallucination_rate: float
    mean_faithfulness: float
    mean_answer_correctness: float
    grounded_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    per_query: list[QueryEvaluationResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
