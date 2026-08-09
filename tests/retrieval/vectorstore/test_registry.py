from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.qdrant_store import QdrantVectorStore
from src.retrieval.vectorstore.registry import (
    UnknownVectorStoreError,
    available_stores,
    get_vector_store,
)


def test_available_stores_lists_all_five() -> None:
    stores = available_stores()
    for name in ("faiss", "chroma", "qdrant", "weaviate", "pinecone"):
        assert name in stores


def test_get_vector_store_faiss(tmp_path: Path) -> None:
    store = get_vector_store("faiss", persist_dir=tmp_path / "f", dimensions=3)
    assert isinstance(store, FaissVectorStore)


def test_get_vector_store_chroma(tmp_path: Path) -> None:
    store = get_vector_store("chroma", persist_dir=tmp_path / "c")
    assert isinstance(store, ChromaVectorStore)


def test_get_vector_store_qdrant(tmp_path: Path) -> None:
    store = get_vector_store("qdrant", persist_dir=tmp_path / "q", dimensions=3)
    assert isinstance(store, QdrantVectorStore)


def test_get_vector_store_unknown_raises() -> None:
    with pytest.raises(UnknownVectorStoreError):
        get_vector_store("not_a_real_store")
