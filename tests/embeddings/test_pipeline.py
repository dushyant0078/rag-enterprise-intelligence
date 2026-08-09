from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.chunking.schemas import Chunk
from src.embeddings.caching import EmbeddingCache
from src.embeddings.pipeline import EmbeddingPipeline
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.utils.hashing import sha256_text


def _make_chunks(texts: list[str], doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}__test__{i:04d}",
            doc_id=doc_id,
            strategy="test",
            chunk_index=i,
            text=text,
            content_sha256=sha256_text(text),
            char_count=len(text),
            token_count_estimate=max(1, len(text) // 4),
        )
        for i, text in enumerate(texts)
    ]


_TEXTS = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "The stock market fell today.",
    "Interest rates are rising.",
]


@pytest.fixture()
def fitted_model() -> TfidfEmbedding:
    return TfidfEmbedding(svd_dimensions=3).fit(_TEXTS)


def test_embed_chunks_shape_and_metadata(fitted_model: TfidfEmbedding) -> None:
    chunks = _make_chunks(_TEXTS)
    pipeline = EmbeddingPipeline(model=fitted_model, batch_size=2)
    embedded = pipeline.embed_chunks(chunks)

    assert len(embedded) == len(chunks)
    for e, c in zip(embedded, chunks, strict=True):
        assert e.chunk_id == c.chunk_id
        assert e.doc_id == c.doc_id
        assert e.model_name == fitted_model.model_name
        assert len(e.vector) == e.dimensions


def test_embed_chunks_uses_cache_on_second_call(
    fitted_model: TfidfEmbedding, tmp_path: Path
) -> None:
    chunks = _make_chunks(_TEXTS)
    cache = EmbeddingCache(tmp_path / "cache")
    pipeline = EmbeddingPipeline(model=fitted_model, cache=cache, batch_size=2)

    first = pipeline.embed_chunks(chunks)
    second = pipeline.embed_chunks(chunks)

    for a, b in zip(first, second, strict=True):
        np.testing.assert_allclose(a.vector, b.vector)


def test_cache_hit_skips_model_call(fitted_model: TfidfEmbedding, tmp_path: Path) -> None:
    chunks = _make_chunks(_TEXTS[:2])
    cache = EmbeddingCache(tmp_path / "cache")
    pipeline = EmbeddingPipeline(model=fitted_model, cache=cache, batch_size=10)
    pipeline.embed_chunks(chunks)  # warms the cache

    call_count = {"n": 0}
    original_embed = fitted_model.embed

    def counting_embed(texts):
        call_count["n"] += 1
        return original_embed(texts)

    fitted_model.embed = counting_embed  # type: ignore[method-assign]
    pipeline.embed_chunks(chunks)
    assert call_count["n"] == 0  # everything was a cache hit — model.embed never called


@pytest.mark.asyncio
async def test_embed_chunks_async_matches_sync(fitted_model: TfidfEmbedding) -> None:
    chunks = _make_chunks(_TEXTS)
    pipeline = EmbeddingPipeline(model=fitted_model, batch_size=2)

    sync_result = pipeline.embed_chunks(chunks)
    async_result = await pipeline.embed_chunks_async(chunks)

    for a, b in zip(sync_result, async_result, strict=True):
        np.testing.assert_allclose(a.vector, b.vector)


def test_embed_and_persist_writes_matching_files(
    fitted_model: TfidfEmbedding, tmp_path: Path
) -> None:
    chunks = _make_chunks(_TEXTS)
    pipeline = EmbeddingPipeline(model=fitted_model)
    npy_path, json_path = pipeline.embed_and_persist(chunks, tmp_path / "out" / "vectors")

    assert npy_path.exists()
    assert json_path.exists()

    import json as json_module

    matrix = np.load(npy_path)
    meta = json_module.loads(json_path.read_text())
    assert matrix.shape[0] == len(chunks)
    assert len(meta) == len(chunks)
    assert meta[0]["chunk_id"] == chunks[0].chunk_id


def test_empty_chunk_list_returns_empty(fitted_model: TfidfEmbedding) -> None:
    pipeline = EmbeddingPipeline(model=fitted_model)
    assert pipeline.embed_chunks([]) == []
