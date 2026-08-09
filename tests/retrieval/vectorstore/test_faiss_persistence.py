from __future__ import annotations

from pathlib import Path

from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord


def test_persist_and_reload(tmp_path: Path) -> None:
    persist_dir = tmp_path / "faiss"
    store = FaissVectorStore(persist_dir=persist_dir, dimensions=3)
    store.upsert(
        [VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={"domain": "hr"})], namespace="ns1"
    )
    store.persist()

    reloaded = FaissVectorStore(persist_dir=persist_dir, dimensions=3)
    assert reloaded.count("ns1") == 1
    results = reloaded.search([1.0, 0.0, 0.0], top_k=1, namespace="ns1")
    assert results[0].id == "a"
    assert results[0].metadata == {"domain": "hr"}


def test_backup_copies_persisted_state(tmp_path: Path) -> None:
    persist_dir = tmp_path / "faiss"
    dest_dir = tmp_path / "backup"
    store = FaissVectorStore(persist_dir=persist_dir, dimensions=3)
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={})], namespace="ns1")

    result_path = store.backup(dest_dir)

    assert result_path == dest_dir
    restored = FaissVectorStore(persist_dir=dest_dir, dimensions=3)
    assert restored.count("ns1") == 1
