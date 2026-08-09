from __future__ import annotations

from src.monitoring.cost_tracking import CostCalculator, ModelPricing
from src.monitoring.token_tracking import TokenUsageTracker


def test_token_tracker_aggregates_by_model() -> None:
    tracker = TokenUsageTracker()
    tracker.record("claude-sonnet-4-5", 1000, 200)
    tracker.record("claude-sonnet-4-5", 500, 100)
    tracker.record("gpt-4.1", 300, 50)

    totals = tracker.total_by_model()
    assert totals["claude-sonnet-4-5"] == {"input_tokens": 1500, "output_tokens": 300}
    assert totals["gpt-4.1"] == {"input_tokens": 300, "output_tokens": 50}


def test_token_tracker_skips_none_counts() -> None:
    tracker = TokenUsageTracker()
    tracker.record("deterministic-test-llm", None, None)
    assert tracker.total_requests() == 0
    assert tracker.total_by_model() == {}


def test_token_tracker_totals() -> None:
    tracker = TokenUsageTracker()
    tracker.record("m", 100, 50)
    tracker.record("m", 100, 50)
    assert tracker.total_requests() == 2
    assert tracker.total_tokens() == 300


def test_cost_calculator_known_model() -> None:
    calc = CostCalculator(
        {"m": ModelPricing(input_cost_per_million=2.0, output_cost_per_million=10.0)}
    )
    est = calc.estimate("m", input_tokens=1_000_000, output_tokens=500_000)
    assert est.pricing_known is True
    assert est.input_cost_usd == 2.0
    assert est.output_cost_usd == 5.0
    assert est.total_cost_usd == 7.0


def test_cost_calculator_unknown_model_returns_zero_not_a_guess() -> None:
    calc = CostCalculator({})
    est = calc.estimate("mystery-model", 1000, 500)
    assert est.pricing_known is False
    assert est.total_cost_usd == 0.0


def test_cost_calculator_estimate_total_across_models() -> None:
    calc = CostCalculator(
        {
            "a": ModelPricing(input_cost_per_million=1.0, output_cost_per_million=1.0),
            "b": ModelPricing(input_cost_per_million=1.0, output_cost_per_million=1.0),
        }
    )
    usage = {
        "a": {"input_tokens": 1_000_000, "output_tokens": 0},
        "b": {"input_tokens": 1_000_000, "output_tokens": 0},
    }
    assert calc.estimate_total(usage) == 2.0
