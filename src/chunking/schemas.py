"""Data model produced by every chunker in src/chunking/."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable unit of text, tagged with enough provenance to build a
    citation (doc_id + page_numbers) and enough bookkeeping to embed and
    index it efficiently (content_sha256 for cache/dedup keys).
    """

    chunk_id: str
    doc_id: str
    strategy: str
    chunk_index: int
    text: str
    content_sha256: str
    char_count: int
    token_count_estimate: int
    page_numbers: list[int] = Field(default_factory=list)
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)  # strategy-specific extras,
    # e.g. {"heading_path": "AML Policy > Enhanced Due Diligence"} for markdown-aware chunks
