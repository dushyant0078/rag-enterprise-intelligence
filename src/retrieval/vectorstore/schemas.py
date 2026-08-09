"""Data model shared by every vector store backend in src/retrieval/vectorstore/."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """One vector to upsert. `metadata` carries everything retrieval needs
    back without a second lookup -- text, doc_id, domain, page_numbers,
    is_table -- so a search result is immediately usable as a RetrievedChunk
    (see src/retrieval/schemas.py) with no join required.
    """

    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
