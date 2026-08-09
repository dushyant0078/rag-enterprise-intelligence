"""ChromaDB vector store backend, using its fully local `PersistentClient`
(writes to a plain directory on disk, no server process) — real and tested
end-to-end in this sandbox, like the FAISS backend. Namespace maps directly
onto a Chroma collection, which supports metadata filtering natively via
`where=`, so no brute-force post-filtering is needed here unlike FAISS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from src.retrieval.vectorstore.base import DEFAULT_NAMESPACE, BaseVectorStore
from src.retrieval.vectorstore.schemas import SearchResult, VectorRecord


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir: Path = Path(
            persist_dir
        )  # narrows base's Path | None -- always local-disk-backed here
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

    def _collection(self, namespace: str):
        return self._client.get_or_create_collection(namespace)

    def upsert(self, records: list[VectorRecord], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not records:
            return
        collection = self._collection(namespace)
        collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            metadatas=[r.metadata or {"_empty": True} for r in records],  # Chroma rejects {}
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str = DEFAULT_NAMESPACE,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        collection = self._collection(namespace)
        if collection.count() == 0:
            return []
        where = self._build_where(filters)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["metadatas", "distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]

        return [
            SearchResult(
                id=id_,
                # Chroma's default space is L2 on (by convention here) normalized vectors;
                # converting to a similarity score keeps this consistent with the other
                # backends, where higher == more similar.
                score=1.0 - (distance / 2.0),
                metadata={k: v for k, v in (metadata or {}).items() if k != "_empty"},
            )
            for id_, distance, metadata in zip(ids, distances, metadatas, strict=True)
        ]

    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not ids:
            return
        self._collection(namespace).delete(ids=ids)

    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        return self._collection(namespace).count()

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: {"$in": value}} if isinstance(value, list) else {key: value}
        # Chroma requires an explicit $and for multi-key equality filters.
        return {
            "$and": [
                {key: ({"$in": value} if isinstance(value, list) else value)}
                for key, value in filters.items()
            ]
        }
