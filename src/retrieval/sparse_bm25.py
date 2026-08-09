"""BM25 sparse retrieval (rank_bm25). Built from the same VectorRecord list
used to populate a vector store, so a caller can feed one records list to
both an embedding-based index and this lexical one — see
scripts/build_index.py.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever, matches_filters
from src.retrieval.schemas import RetrievedChunk
from src.retrieval.vectorstore.schemas import VectorRecord
from src.utils.text import tokenize


class BM25Retriever(BaseRetriever):
    retriever_name = "sparse_bm25"

    def __init__(self, records: list[VectorRecord]) -> None:
        self._records = {r.id: r for r in records}
        self._ids = [r.id for r in records]
        tokenized_corpus = [tokenize(r.metadata.get("text", "")) for r in records]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda pair: -pair[1])

        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            if score <= 0:
                break  # rank_bm25 sorts descending — once we hit zero, nothing further matches
            metadata = dict(self._records[chunk_id].metadata)
            if not matches_filters(metadata, filters):
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=metadata.pop("doc_id", ""),
                    text=metadata.pop("text", ""),
                    score=float(score),
                    domain=metadata.pop("domain", None),
                    page_numbers=metadata.pop("page_numbers", []),
                    is_table=metadata.pop("is_table", False),
                    metadata=metadata,
                    source_retriever=self.retriever_name,
                )
            )
            if len(results) >= top_k:
                break
        return results
