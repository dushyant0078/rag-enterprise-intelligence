"""DeepEval integration.

Unlike RAGAS (see ragas_integration.py), `deepeval` imports cleanly in this
sandbox, and `LLMTestCase` construction works with no LLM involved --
verified directly (see build_test_case). Metric *construction* itself,
though, not just `.measure()`, requires a configured LLM judge:
`FaithfulnessMetric()` with no credentials raises immediately
(`deepeval.errors.DeepEvalError: OpenAI API key is not configured`) before
any evaluation is even attempted -- confirmed by actually calling it here,
not assumed from documentation. No LLM is reachable end-to-end in this
sandbox with valid credentials (ClaudeLLM reaches the real API but this
environment has no API key -- see docs/07_conversational_rag_pipeline.md).
This module is real, correct integration code, verified as far as it can be
without a judge model; running an actual evaluation is the one remaining
step once one is available.
"""

from __future__ import annotations

from typing import Any

from src.citations.schemas import ValidatedAnswer
from src.evaluation.schemas import BenchmarkQuery


def build_test_case(benchmark_query: BenchmarkQuery, result: ValidatedAnswer) -> Any:
    """Build a deepeval LLMTestCase from one benchmark query's pipeline
    result -- the same shape a real deepeval evaluation run would consume.
    """
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=benchmark_query.query,
        actual_output=result.answer_text,
        expected_output=benchmark_query.expected_answer,
        retrieval_context=[c.text for c in result.retrieved_passages],
    )


def build_metrics(threshold: float = 0.7, model: str | None = None) -> list[Any]:
    """Construct the deepeval metric suite covering the spec's named
    generation metrics: Faithfulness, Answer Relevancy, Context Precision,
    Context Recall, and Hallucination. `model` is the LLM judge identifier
    deepeval should use -- left to the caller since it depends on which
    credentials are actually configured in the deployment.

    Raises deepeval.errors.DeepEvalError immediately if no LLM judge
    credentials are configured -- construction itself needs a judge, not
    just measure(). Confirmed directly: calling this with no `model` and no
    OPENAI_API_KEY set fails at `FaithfulnessMetric(**kwargs)` with exactly
    that error, not a generic import or type error.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )

    kwargs: dict[str, Any] = {"threshold": threshold}
    if model:
        kwargs["model"] = model

    return [
        FaithfulnessMetric(**kwargs),
        AnswerRelevancyMetric(**kwargs),
        ContextualPrecisionMetric(**kwargs),
        ContextualRecallMetric(**kwargs),
        HallucinationMetric(**kwargs),
    ]


def evaluate_with_deepeval(
    benchmark_query: BenchmarkQuery, result: ValidatedAnswer, *, model: str | None = None
) -> dict[str, float]:
    """Run the full deepeval metric suite against one query's result.
    Requires a configured LLM judge (via `model` or deepeval's own default
    credential resolution) -- will raise if none is available. Not called
    anywhere in this project's own test suite or demo scripts for exactly
    that reason; see RAGEvaluator's lexical-proxy metrics for what runs here
    instead.
    """
    test_case = build_test_case(benchmark_query, result)
    metrics = build_metrics(model=model)

    scores: dict[str, float] = {}
    for metric in metrics:
        metric.measure(test_case)
        scores[metric.__class__.__name__] = metric.score
    return scores
