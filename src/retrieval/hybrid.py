"""Hybrid retrieval: weighted fusion of dense and BM25 sparse results. Each
result set is min-max normalized to [0, 1] before combining — dense cosine
similarity and BM25's unbounded scores live on completely different scales,
so combining them raw would let whichever retriever happens to produce
bigger numbers dominate regardless of actual relevance.

Default weights (0.6 dense / 0.4 sparse) match
configs/retrieval/hybrid.yaml — this is the strategy that config selects.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk


def _min_max_normalize(results: list[RetrievedChunk]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {r.chunk_id: 1.0 for r in results}
    return {r.chunk_id: (r.score - lo) / (hi - lo) for r in results}


class HybridRetriever(BaseRetriever):
    retriever_name = "hybrid"

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        candidate_multiplier: int = 3,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        fetch_k = top_k * self.candidate_multiplier
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k, filters=filters)

        dense_norm = _min_max_normalize(dense_results)
        sparse_norm = _min_max_normalize(sparse_results)

        chunk_lookup: dict[str, RetrievedChunk] = {r.chunk_id: r for r in dense_results}
        for r in sparse_results:
            chunk_lookup.setdefault(r.chunk_id, r)

        fused_scores = {
            chunk_id: self.dense_weight * dense_norm.get(chunk_id, 0.0)
            + self.sparse_weight * sparse_norm.get(chunk_id, 0.0)
            for chunk_id in chunk_lookup
        }

        ranked_ids = sorted(fused_scores, key=lambda cid: -fused_scores[cid])[:top_k]
        return [
            chunk_lookup[cid].model_copy(
                update={"score": fused_scores[cid], "source_retriever": self.retriever_name}
            )
            for cid in ranked_ids
        ]
