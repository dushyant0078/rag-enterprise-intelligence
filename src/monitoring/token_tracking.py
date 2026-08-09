"""Token usage tracking: aggregates the input/output token counts every
LLMResponse already carries (Milestone 7's BaseLLM contract) -- no LLM call
needed to test this, it's pure aggregation over numbers already produced
elsewhere in the pipeline.
"""

from __future__ import annotations

from collections import defaultdict

from src.monitoring.schemas import TokenUsageRecord


class TokenUsageTracker:
    def __init__(self) -> None:
        self._records: list[TokenUsageRecord] = []

    def record(self, model: str, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is None or output_tokens is None:
            return  # some backends (e.g. DeterministicTestLLM) don't report token counts at all
        self._records.append(
            TokenUsageRecord(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        )

    def total_by_model(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0}
        )
        for r in self._records:
            totals[r.model]["input_tokens"] += r.input_tokens
            totals[r.model]["output_tokens"] += r.output_tokens
        return dict(totals)

    def total_requests(self) -> int:
        return len(self._records)

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self._records)
