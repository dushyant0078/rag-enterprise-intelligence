"""Data model produced by the embedding pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class EmbeddedChunk(BaseModel):
    """A Chunk (see src/chunking/schemas.py) plus its embedding vector and
    the provenance needed to know whether that vector is still valid (model
    name + dimensions — a config change invalidates cached vectors).
    """

    chunk_id: str
    doc_id: str
    vector: list[float]
    model_name: str
    dimensions: int
    normalized: bool
