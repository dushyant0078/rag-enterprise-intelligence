"""Data model returned by every LLM backend in src/llm/."""

from __future__ import annotations

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
