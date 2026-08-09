"""One test suite, run against FAISS, Chroma, and Qdrant identically —
directly validates the "interchangeable vector store" promise: if these
pass for all three params, the abstraction actually holds, not just for
whichever backend happened to get hand-tested.

Weaviate and Pinecone aren't parametrized in here (no reachable service in
this sandbox — see their modules' docstrings); test_registry.py separately
confirms the registry wires them correctly without requiring a live service.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.vectorstore.base import BaseVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord

_BACKENDS = ("faiss", "chroma", "qdrant")


@pytest.fixture(params=_BACKENDS)
def store(request, tmp_path: Path) -> BaseVectorStore:
    backend = request.param
    persist_dir = tmp_path / backend
    if backend == "faiss":
        from src.retrieval.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(persist_dir=persist_dir, dimensions=3)
    if backend == "chroma":
        from src.retrieval.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(persist_dir=persist_dir)
    if backend == "qdrant":
        from src.retrieval.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(persist_dir=persist_dir, dimensions=3)
    raise ValueError(backend)


def _sample_records() -> list[VectorRecord]:
    return [
        VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={"domain": "hr", "is_table": False}),
        VectorRecord(
            id="b", vector=[0.0, 1.0, 0.0], metadata={"domain": "legal", "is_table": False}
        ),
        VectorRecord(id="c", vector=[0.9, 0.1, 0.0], metadata={"domain": "hr", "is_table": True}),
    ]


def test_upsert_and_count(store: BaseVectorStore) -> None:
    store.upsert(_sample_records(), namespace="ns1")
    assert store.count("ns1") == 3


def test_search_returns_closest_first(store: BaseVectorStore) -> None:
    store.upsert(_sample_records(), namespace="ns1")
    results = store.search([1.0, 0.0, 0.0], top_k=3, namespace="ns1")
    assert results[0].id == "a"
    assert results[0].score > results[1].score > results[2].score
    assert results[-1].id == "b"  # orthogonal vector — least similar


def test_search_respects_top_k(store: BaseVectorStore) -> None:
    store.upsert(_sample_records(), namespace="ns1")
    results = store.search([1.0, 0.0, 0.0], top_k=1, namespace="ns1")
    assert len(results) == 1


def test_metadata_filter_equality(store: BaseVectorStore) -> None:
    store.upsert(_sample_records(), namespace="ns1")
    results = store.search([1.0, 0.0, 0.0], top_k=3, namespace="ns1", filters={"domain": "legal"})
    assert [r.id for r in results] == ["b"]


def test_metadata_filter_list_membership(store: BaseVectorStore) -> None:
    store.upsert(_sample_records(), namespace="ns1")
    results = store.search(
        [1.0, 0.0, 0.0], top_k=3, namespace="ns1", filters={"domain": ["legal", "hr"]}
    )
    assert {r.id for r in results} == {"a", "b", "c"}


def test_namespace_isolation(store: BaseVectorStore) -> None:
    store.upsert([VectorRecord(id="x", vector=[1.0, 0.0, 0.0], metadata={})], namespace="ns_a")
    store.upsert([VectorRecord(id="y", vector=[1.0, 0.0, 0.0], metadata={})], namespace="ns_b")
    assert store.count("ns_a") == 1
    assert store.count("ns_b") == 1
    results_a = store.search([1.0, 0.0, 0.0], top_k=10, namespace="ns_a")
    assert [r.id for r in results_a] == ["x"]


def test_delete(store: BaseVectorStore) -> None:
    store.upsert(_sample_records(), namespace="ns1")
    store.delete(["a"], namespace="ns1")
    assert store.count("ns1") == 2
    results = store.search([1.0, 0.0, 0.0], top_k=3, namespace="ns1")
    assert "a" not in {r.id for r in results}


def test_upsert_overwrites_existing_id(store: BaseVectorStore) -> None:
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={"v": 1})], namespace="ns1")
    store.upsert([VectorRecord(id="a", vector=[0.0, 1.0, 0.0], metadata={"v": 2})], namespace="ns1")
    assert store.count("ns1") == 1
    results = store.search([0.0, 1.0, 0.0], top_k=1, namespace="ns1")
    assert results[0].metadata["v"] == 2


def test_search_on_empty_namespace_returns_empty(store: BaseVectorStore) -> None:
    assert store.search([1.0, 0.0, 0.0], top_k=5, namespace="never_used") == []


def test_count_on_empty_namespace_is_zero(store: BaseVectorStore) -> None:
    assert store.count("never_used") == 0
