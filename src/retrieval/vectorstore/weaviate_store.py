"""Weaviate vector store backend.

Unlike FAISS/Chroma/Qdrant, Weaviate has no fully-offline embedded mode that
avoids both a running server *and* a network download — its "embedded"
option still fetches a server binary on first run, and this sandbox's
network allowlist doesn't reach the hosts that would come from. This class
is correct, complete code against the real `weaviate-client` v4 API; it
needs a reachable Weaviate instance (local Docker container or Weaviate
Cloud) to actually run — see docs/05_vectorstore_and_retrieval.md for the
same "written and correct, not verified here" treatment already applied to
SentenceTransformerEmbedding in Milestone 4.
"""

from __future__ import annotations

from typing import Any

import weaviate
import weaviate.classes as wvc

from src.retrieval.vectorstore.base import DEFAULT_NAMESPACE, BaseVectorStore
from src.retrieval.vectorstore.schemas import SearchResult, VectorRecord


def _collection_name(namespace: str) -> str:
    # Weaviate class/collection names must start with an uppercase letter.
    return f"Ns_{namespace}"


class WeaviateVectorStore(BaseVectorStore):
    def __init__(self, url: str, api_key: str | None = None, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self._client = weaviate.connect_to_custom(
            http_host=url, http_port=8080, http_secure=False,
            grpc_host=url, grpc_port=50051, grpc_secure=False,
            auth_credentials=auth,
        )  # fmt: skip

    def _ensure_collection(self, namespace: str):
        name = _collection_name(namespace)
        if not self._client.collections.exists(name):
            self._client.collections.create(
                name=name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
        return self._client.collections.get(name)

    def upsert(self, records: list[VectorRecord], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not records:
            return
        collection = self._ensure_collection(namespace)
        with collection.batch.dynamic() as batch:
            for r in records:
                batch.add_object(
                    properties={**r.metadata, "_chunk_id": r.id},
                    vector=r.vector,
                    uuid=_deterministic_uuid(r.id),
                )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str = DEFAULT_NAMESPACE,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        name = _collection_name(namespace)
        if not self._client.collections.exists(name):
            return []
        collection = self._client.collections.get(name)
        weaviate_filter = _build_filter(filters)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )
        results = []
        for obj in response.objects:
            props = dict(obj.properties)
            chunk_id = props.pop("_chunk_id", str(obj.uuid))
            if not isinstance(chunk_id, str):
                chunk_id = str(chunk_id)
            distance = obj.metadata.distance or 0.0
            results.append(SearchResult(id=chunk_id, score=1.0 - distance, metadata=props))
        return results

    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> None:
        name = _collection_name(namespace)
        if not ids or not self._client.collections.exists(name):
            return
        collection = self._client.collections.get(name)
        for chunk_id in ids:
            collection.data.delete_by_id(_deterministic_uuid(chunk_id))

    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        name = _collection_name(namespace)
        if not self._client.collections.exists(name):
            return 0
        total = self._client.collections.get(name).aggregate.over_all(total_count=True).total_count
        return total or 0

    def close(self) -> None:
        self._client.close()


def _deterministic_uuid(chunk_id: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _build_filter(filters: dict[str, Any] | None):
    if not filters:
        return None
    conditions = [wvc.query.Filter.by_property(k).equal(v) for k, v in filters.items()]
    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined & cond
    return combined
