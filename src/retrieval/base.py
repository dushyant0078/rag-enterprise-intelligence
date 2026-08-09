"""Base retriever interface. Every strategy in src/retrieval/ implements
this — dense, sparse, and every retriever that wraps another retriever
(hybrid, multi-query, ensemble, ...) all compose through this one contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.retrieval.schemas import RetrievedChunk


class BaseRetriever(ABC):
    retriever_name: str = "base"

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks relevant to `query`, optionally
        restricted by equality/list-membership metadata filters (the same
        filter contract as BaseVectorStore.search).
        """
        raise NotImplementedError


def matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Shared filter-matching logic for retrievers that filter in Python
    (BM25, and any wrapper re-checking after fusing multiple result sets) —
    same semantics as the vector store backends' native filtering.
    """
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
