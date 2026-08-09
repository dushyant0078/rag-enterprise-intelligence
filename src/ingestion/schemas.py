"""Data model for what ingestion writes to data/metadata/ and returns to callers."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class IngestionStatus(StrEnum):
    INGESTED = "ingested"
    NEW_VERSION = "new_version"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    DUPLICATE = "duplicate"
    ERROR = "error"


class DocumentMetadata(BaseModel):
    """Everything downstream stages (chunking, RBAC-aware retrieval, citations)
    need to know about one ingested document, independent of its parsed content.
    """

    doc_id: str
    source_path: str
    domain: str  # namespace used for RBAC/retrieval isolation — see docs/02_architecture.md #8
    format: str
    sha256: str
    content_sha256: str
    size_bytes: int
    page_count: int
    language: str | None = None
    version: int = 1
    is_duplicate_of: str | None = None
    used_ocr: bool = False
    parser_warnings: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # populated where determinable; drives recency-aware retrieval later
    effective_date: str | None = None


class IngestionResult(BaseModel):
    doc_id: str
    status: IngestionStatus
    metadata: DocumentMetadata | None = None
    message: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
