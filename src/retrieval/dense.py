"""Dense retrieval: embed the query, search a vector store. The most basic
strategy — every other retriever in this package either wraps this one or
composes with it.
"""

from __future__ import annotations

from typing import Any

from src.embeddings.base import EmbeddingModel
from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk
from src.retrieval.vectorstore.base import DEFAULT_NAMESPACE, BaseVectorStore
from src.retrieval.vectorstore.schemas import SearchResult


def search_result_to_retrieved_chunk(result: SearchResult, source_retriever: str) -> RetrievedChunk:
    """Shared conversion from a raw vector-store SearchResult to a
    RetrievedChunk. Expects the indexing step to have put `text`, `doc_id`,
    `domain`, `page_numbers`, and `is_table` into the record's metadata
    payload at upsert time (see scripts/build_index.py) — everything else in
    metadata rides along in RetrievedChunk.metadata untouched.
    """
    metadata = dict(result.metadata)
    return RetrievedChunk(
        chunk_id=result.id,
        doc_id=metadata.pop("doc_id", ""),
        text=metadata.pop("text", ""),
        score=result.score,
        domain=metadata.pop("domain", None),
        page_numbers=metadata.pop("page_numbers", []),
        is_table=metadata.pop("is_table", False),
        metadata=metadata,
        source_retriever=source_retriever,
    )


class DenseRetriever(BaseRetriever):
    retriever_name = "dense"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: BaseVectorStore,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.namespace = namespace

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        query_vector = self.embedding_model.embed([query])[0]
        results = self.vector_store.search(
            query_vector.tolist(), top_k=top_k, namespace=self.namespace, filters=filters
        )
        return [search_result_to_retrieved_chunk(r, self.retriever_name) for r in results]
