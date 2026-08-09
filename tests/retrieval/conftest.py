from __future__ import annotations

from typing import Any

import pytest

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk


class FakeRetriever(BaseRetriever):
    """Test double: returns a pre-programmed result list for any query,
    optionally applying filters itself so wrapper-retriever tests (hybrid,
    ensemble, metadata_aware, ...) can be verified in isolation without
    standing up a real vector store or embedding model for every case.
    """

    retriever_name = "fake"

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, dict[str, Any] | None]] = []

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, filters))
        results = self._results
        if filters:
            results = [
                r
                for r in results
                if all(
                    (
                        (r.domain in v if isinstance(v, list) else r.domain == v)
                        if k == "domain"
                        else r.metadata.get(k) == v
                    )
                    for k, v in filters.items()
                )
            ]
        return results[:top_k]


def make_chunk(
    chunk_id: str,
    score: float,
    *,
    doc_id: str = "doc1",
    domain: str = "hr",
    text: str = "some chunk text",
    metadata: dict[str, Any] | None = None,
    is_table: bool = False,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        score=score,
        domain=domain,
        is_table=is_table,
        metadata=metadata or {},
    )


@pytest.fixture()
def make_retrieved_chunk():
    return make_chunk
