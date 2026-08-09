"""Cost estimation from token usage.

**Pricing is a point-in-time snapshot, not a live lookup** -- LLM API
pricing changes on providers' own schedules, and this project has no
network path to any pricing API (nor does one really exist). The table
below uses each provider's approximate, publicly published per-million-token
rates as of this project's training data; a real deployment should treat
`DEFAULT_PRICING` as a starting point to verify and keep current, not as a
guaranteed-accurate source -- `CostCalculator` accepts a custom pricing
table specifically so that's a config change, not a code change. A model
not in the table returns cost 0.0 with `pricing_known=False`, never a
silently wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.monitoring.schemas import CostEstimate


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_million: float
    output_cost_per_million: float


# Approximate USD per 1M tokens, as of this project's training data. Verify against each
# provider's current pricing page before using this for real budget decisions.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-5": ModelPricing(input_cost_per_million=3.00, output_cost_per_million=15.00),
    "gpt-4.1": ModelPricing(input_cost_per_million=2.00, output_cost_per_million=8.00),
    "gemini-2.5-pro": ModelPricing(input_cost_per_million=1.25, output_cost_per_million=10.00),
}


class CostCalculator:
    def __init__(self, pricing: dict[str, ModelPricing] | None = None) -> None:
        self.pricing = pricing if pricing is not None else DEFAULT_PRICING

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
        rates = self.pricing.get(model)
        if rates is None:
            return CostEstimate(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
                pricing_known=False,
            )

        input_cost = (input_tokens / 1_000_000) * rates.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * rates.output_cost_per_million
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(input_cost + output_cost, 6),
            pricing_known=True,
        )

    def estimate_total(self, usage_by_model: dict[str, dict[str, int]]) -> float:
        total = 0.0
        for model, counts in usage_by_model.items():
            estimate = self.estimate(model, counts["input_tokens"], counts["output_tokens"])
            total += estimate.total_cost_usd
        return round(total, 6)
