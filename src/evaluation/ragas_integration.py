"""RAGAS integration.

**Import unverified in this sandbox, for a different and stronger reason
than every other "untestable" backend in this project: a genuine,
reproducible upstream packaging conflict, not a network restriction.**

`pip install ragas` succeeds, but `import ragas` fails --
`ragas.llms.base` unconditionally imports
`langchain_community.chat_models.vertexai`, which doesn't exist in the
`langchain-community` version ragas's own dependency resolution installs
alongside it (confirmed directly: `ModuleNotFoundError:
No module named 'langchain_community.chat_models.vertexai'`). Attempting to
downgrade `langchain-community` to a version that still has that module
breaks `langchain-openai`'s requirement on a newer `langchain-core`
(`ImportError: cannot import name 'ContextOverflowError'`) -- a real,
current three-way version conflict across ragas/langchain-community/
langchain-openai in the PyPI ecosystem as of this writing, not a mistake in
this project's setup.

Because of that, the functions below are written directly from RAGAS's
documented API (metric classes, `evaluate()`, `EvaluationDataset`) rather
than verified against a working import here. Treat this module as a
starting point to adjust once a compatible dependency set is pinned (or a
newer ragas release fixes the conflict), not as tested integration code the
way deepeval_integration.py is.
"""

from __future__ import annotations

from typing import Any, cast

from src.citations.schemas import ValidatedAnswer
from src.evaluation.benchmark_dataset import BenchmarkDataset


def build_evaluation_dataset(pairs: list[tuple[Any, ValidatedAnswer]]) -> Any:
    """Build a ragas EvaluationDataset from (BenchmarkQuery, ValidatedAnswer)
    pairs. Written from RAGAS's documented `SingleTurnSample` /
    `EvaluationDataset` schema -- unverified here (see module docstring).
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=query.query,
            response=result.answer_text,
            reference=query.expected_answer,
            retrieved_contexts=[c.text for c in result.retrieved_passages],
        )
        for query, result in pairs
    ]
    return EvaluationDataset(samples=cast(list, samples))


def evaluate_with_ragas(
    dataset: BenchmarkDataset, results: list[ValidatedAnswer], *, llm: Any
) -> Any:
    """Run RAGAS's standard metric suite (Faithfulness, AnswerCorrectness,
    ContextPrecision, ContextRecall -- the spec's named metrics) against a
    benchmark run. `llm` must be a RAGAS-compatible LLM wrapper (RAGAS
    expects a LangChain-style interface, so this project's own BaseLLM
    subclasses -- ClaudeLLM included -- would need a thin adapter, not a
    direct pass-through; writing that adapter is the natural next step once
    the import conflict above is resolved and a live judge call can
    actually be tested).
    """
    from ragas import evaluate
    from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

    eval_dataset = build_evaluation_dataset(list(zip(dataset.queries, results, strict=True)))
    return evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), AnswerCorrectness(), ContextPrecision(), ContextRecall()],
        llm=llm,
    )
