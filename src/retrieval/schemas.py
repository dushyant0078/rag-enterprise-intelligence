"""Data model returned by every retriever in src/retrieval/."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A retrieved chunk, ready to hand to the Context Builder /
    Citation Generator (docs/02_architecture.md) — everything needed for a
    citation (doc_id, page_numbers) travels with it, no second lookup needed.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    domain: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_retriever: str = "unknown"
