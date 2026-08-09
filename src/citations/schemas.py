"""Data model for src/citations/: every citation an LLM's answer makes gets
resolved against the chunks actually provided as context (or flagged as
fabricated if it doesn't), and the validated answer carries every field the
spec's citation-aware-answering section requires: source document, page
number, chunk ID, confidence score, and retrieved passages.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.retrieval.schemas import RetrievedChunk


class Citation(BaseModel):
    doc_id: str
    page_number: int | None = None
    chunk_id: str | None = None  # None means this citation didn't resolve to any provided chunk
    confidence: float = 0.0  # the matched chunk's retrieval/rerank score; 0.0 if unresolved
    excerpt: str = ""  # short excerpt from the matched chunk supporting this citation

    @property
    def is_fabricated(self) -> bool:
        return self.chunk_id is None


class ValidatedAnswer(BaseModel):
    answer_text: str  # final text shown to the user -- may be the fallback phrase, see is_grounded
    original_answer_text: str  # the LLM's raw output, before any validation-driven replacement
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: float = 0.0
    is_grounded: bool = False
    retrieved_passages: list[RetrievedChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def fabricated_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.is_fabricated]
