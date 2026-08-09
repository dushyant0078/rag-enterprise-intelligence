"""Embedding pipeline: wires a chunk list + an EmbeddingModel + an optional
EmbeddingCache together. Handles batching, skips chunks whose vector is
already cached, and offers both a sync and a genuinely-concurrent async path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from src.chunking.schemas import Chunk
from src.embeddings.base import EmbeddingModel
from src.embeddings.caching import EmbeddingCache
from src.embeddings.schemas import EmbeddedChunk
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        model: EmbeddingModel,
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.cache = cache
        self.batch_size = batch_size

    # -- sync -----------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        vectors: list[np.ndarray | None] = [None] * len(chunks)
        pending_idx, pending_texts = self._resolve_cache_hits(chunks, vectors)

        for start in range(0, len(pending_texts), self.batch_size):
            batch_idx = pending_idx[start : start + self.batch_size]
            batch_texts = pending_texts[start : start + self.batch_size]
            batch_vectors = self.model.embed(batch_texts)
            self._store_batch(batch_idx, batch_texts, batch_vectors, vectors)

        logger.info(
            "embedded chunks",
            extra={"total": len(chunks), "cache_hits": len(chunks) - len(pending_texts)},
        )
        return self._assemble(chunks, vectors)

    # -- async ------------------------------------------------------------

    async def embed_chunks_async(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        vectors: list[np.ndarray | None] = [None] * len(chunks)
        pending_idx, pending_texts = self._resolve_cache_hits(chunks, vectors)

        batches = [
            (pending_idx[i : i + self.batch_size], pending_texts[i : i + self.batch_size])
            for i in range(0, len(pending_texts), self.batch_size)
        ]
        results = await asyncio.gather(*(self.model.embed_async(texts) for _, texts in batches))
        for (batch_idx, batch_texts), batch_vectors in zip(batches, results, strict=True):
            self._store_batch(batch_idx, batch_texts, batch_vectors, vectors)

        return self._assemble(chunks, vectors)

    # -- persistence --------------------------------------------------------

    def embed_and_persist(self, chunks: list[Chunk], out_path: str | Path) -> tuple[Path, Path]:
        """Embed `chunks` and write two files: `<out_path>.npy` (stacked
        vector matrix, row order matching the metadata file) and
        `<out_path>.json` (chunk_id/doc_id/model_name per row) — the pairing
        Milestone 5's vector-store loader will read.
        """
        embedded = self.embed_chunks(chunks)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        matrix = np.array([e.vector for e in embedded], dtype=float)
        npy_path = out_path.with_suffix(".npy")
        json_path = out_path.with_suffix(".json")

        np.save(npy_path, matrix)
        json_path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": e.chunk_id,
                        "doc_id": e.doc_id,
                        "model_name": e.model_name,
                        "dimensions": e.dimensions,
                        "normalized": e.normalized,
                    }
                    for e in embedded
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        return npy_path, json_path

    # -- internals --------------------------------------------------------

    def _resolve_cache_hits(
        self, chunks: list[Chunk], vectors: list[np.ndarray | None]
    ) -> tuple[list[int], list[str]]:
        pending_idx: list[int] = []
        pending_texts: list[str] = []
        for i, c in enumerate(chunks):
            cached = self.cache.get(c.text, self.model.model_name) if self.cache else None
            if cached is not None:
                vectors[i] = cached
            else:
                pending_idx.append(i)
                pending_texts.append(c.text)
        return pending_idx, pending_texts

    def _store_batch(
        self,
        batch_idx: list[int],
        batch_texts: list[str],
        batch_vectors: np.ndarray,
        vectors: list[np.ndarray | None],
    ) -> None:
        for idx, text, vec in zip(batch_idx, batch_texts, batch_vectors, strict=True):
            vectors[idx] = vec
            if self.cache:
                self.cache.set(text, self.model.model_name, vec)

    def _assemble(
        self, chunks: list[Chunk], vectors: list[np.ndarray | None]
    ) -> list[EmbeddedChunk]:
        normalized_flag = getattr(self.model, "normalize", True)
        result = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if vector is None:
                raise RuntimeError(f"chunk {chunk.chunk_id} was never embedded — internal error")
            result.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    vector=np.asarray(vector).tolist(),
                    model_name=self.model.model_name,
                    dimensions=self.model.dimensions,
                    normalized=normalized_flag,
                )
            )
        return result
