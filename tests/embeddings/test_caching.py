from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings.caching import EmbeddingCache


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    assert cache.get("some text", "model-a") is None


def test_cache_set_and_get(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    vector = np.array([1.0, 2.0, 3.0])
    cache.set("some text", "model-a", vector)
    result = cache.get("some text", "model-a")
    assert result is not None
    np.testing.assert_array_equal(result, vector)


def test_cache_is_keyed_by_model_name(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.set("text", "model-a", np.array([1.0]))
    assert cache.get("text", "model-b") is None


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    EmbeddingCache(tmp_path).set("text", "model-a", np.array([1.0, 2.0]))
    reloaded = EmbeddingCache(tmp_path)
    result = reloaded.get("text", "model-a")
    assert result is not None
    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


def test_get_many(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.set("a", "model-a", np.array([1.0]))
    results = cache.get_many(["a", "b"], "model-a")
    assert results[0] is not None
    assert results[1] is None
