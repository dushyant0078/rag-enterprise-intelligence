"""Qdrant vector store backend, using its embedded local-path mode
(`QdrantClient(path=...)`) — no server process, fully on-disk — real and
tested end-to-end in this sandbox, like FAISS and Chroma. A production
deployment would point this at a real Qdrant server URL instead (see
configs/vectorstore/qdrant.yaml under Milestone 2 — same client class,
different constructor argument); the retrieval logic doesn't change either
way. Namespace maps onto a Qdrant collection, with native metadata filtering
via `query_filter=`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.retrieval.vectorstore.base import DEFAULT_NAMESPACE, BaseVectorStore
from src.retrieval.vectorstore.schemas import SearchResult, VectorRecord


def _stable_point_id(str_id: str) -> int:
    """Qdrant point ids must be an unsigned int or a UUID, not an arbitrary
    string — deterministically map our chunk_id strings onto one via a
    truncated hash, keeping the original string in the payload for lookups.
    """
    import hashlib

    return int(hashlib.sha256(str_id.encode("utf-8")).hexdigest()[:16], 16)


def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
    if not filters:
        return None
    conditions: list[FieldCondition] = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    # qdrant-client's Filter.must is typed as a broader condition union than list[FieldCondition]
    # alone; list's invariance means passing our narrower list needs an explicit cast even
    # though every element genuinely satisfies the union.
    return Filter(must=cast(list, conditions))


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, persist_dir: str | Path, dimensions: int) -> None:
        self.persist_dir: Path = Path(
            persist_dir
        )  # narrows base's Path | None -- always local-disk-backed here
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self._client = QdrantClient(path=str(self.persist_dir))

    def _ensure_collection(self, namespace: str) -> None:
        if not self._client.collection_exists(namespace):
            self._client.create_collection(
                collection_name=namespace,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    def upsert(self, records: list[VectorRecord], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not records:
            return
        self._ensure_collection(namespace)
        points = [
            PointStruct(
                id=_stable_point_id(r.id),
                vector=r.vector,
                payload={**r.metadata, "_chunk_id": r.id},
            )
            for r in records
        ]
        self._client.upsert(collection_name=namespace, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str = DEFAULT_NAMESPACE,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not self._client.collection_exists(namespace):
            return []
        result = self._client.query_points(
            collection_name=namespace,
            query=query_vector,
            limit=top_k,
            query_filter=_build_filter(filters),
        )
        results = []
        for point in result.points:
            payload = dict(point.payload or {})
            chunk_id = payload.pop("_chunk_id", str(point.id))
            results.append(SearchResult(id=chunk_id, score=float(point.score), metadata=payload))
        return results

    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not ids or not self._client.collection_exists(namespace):
            return
        self._client.delete(
            collection_name=namespace, points_selector=[_stable_point_id(i) for i in ids]
        )

    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        if not self._client.collection_exists(namespace):
            return 0
        return self._client.count(collection_name=namespace).count
