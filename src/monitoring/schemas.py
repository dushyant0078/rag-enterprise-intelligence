"""Data model for src/monitoring/."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TokenUsageRecord(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CostEstimate(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    pricing_known: (
        bool  # False when the model isn't in the pricing table -- cost is 0.0, not a guess
    )


class RetrievalObservation(BaseModel):
    query: str
    result_count: int
    top_score: float | None
    mean_score: float | None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Alert(BaseModel):
    name: str
    severity: str  # "warning" | "critical"
    message: str
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: dict = Field(default_factory=dict)


class DriftReport(BaseModel):
    metric: str
    baseline_size: int
    current_size: int
    statistic: float
    p_value: float
    drifted: bool  # p_value below the configured significance threshold
    threshold: float


class FeedbackRecord(BaseModel):
    feedback_id: str
    session_id: str
    user_id: str
    rating: str  # "up" | "down"
    comment: str | None = None
    query: str | None = None
    answer_excerpt: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
