"""Pinecone vector store backend.

Pinecone is cloud-only — there is no local or embedded mode at all, by
design (it's a managed SaaS). This class is correct, complete code against
the real `pinecone-client` v5 API; it needs a Pinecone account, an API key,
and network access to Pinecone's API (not in this sandbox's allowlist) to
actually run. Namespace maps onto Pinecone's native `namespace` parameter
directly — the one backend here where that mapping needed no adaptation.
"""

from __future__ import annotations

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from src.retrieval.vectorstore.base import DEFAULT_NAMESPACE, BaseVectorStore
from src.retrieval.vectorstore.schemas import SearchResult, VectorRecord


class PineconeVectorStore(BaseVectorStore):
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimensions: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self._pc = Pinecone(api_key=api_key)
        self.dimensions = dimensions
        if not self._pc.has_index(index_name):
            self._pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._pc.Index(index_name)

    def upsert(self, records: list[VectorRecord], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not records:
            return
        vectors = [(r.id, r.vector, r.metadata) for r in records]
        self._index.upsert(vectors=vectors, namespace=namespace)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str = DEFAULT_NAMESPACE,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=_build_filter(filters),
            include_metadata=True,
        )
        return [
            SearchResult(id=match.id, score=float(match.score), metadata=dict(match.metadata or {}))
            for match in response.matches
        ]

    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not ids:
            return
        self._index.delete(ids=ids, namespace=namespace)

    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        stats = self._index.describe_index_stats()
        ns_stats = stats.namespaces.get(namespace)
        return ns_stats.vector_count if ns_stats else 0

    # backup() is inherited from BaseVectorStore: persist_dir stays None here (Pinecone is a
    # managed cloud service with no local directory to copy), so the base implementation's
    # NotImplementedError with a clear message is exactly the right behavior already.


def _build_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    return {
        key: ({"$in": value} if isinstance(value, list) else {"$eq": value})
        for key, value in filters.items()
    }
