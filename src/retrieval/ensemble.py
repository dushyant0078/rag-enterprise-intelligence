"""Ensemble retrieval: Reciprocal Rank Fusion (RRF) across an arbitrary list
of retrievers. Unlike HybridRetriever's score-based fusion (which needs
comparable, normalized scores), RRF only looks at each retriever's *rank
order*, which makes it robust to combining retrievers whose scores aren't on
any common scale at all — e.g. a dense retriever, a BM25 retriever, and a
future reranker-backed retriever (Milestone 6), all three at once.

RRF score for a chunk = sum over retrievers of weight / (k + rank), a
standard, well-established fusion formula (Cormack et al., 2009) — not a
heuristic invented for this project.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk

_DEFAULT_RRF_K = (
    60  # the constant from the original RRF paper; dampens the impact of rank 1 vs rank 2
)


class EnsembleRetriever(BaseRetriever):
    retriever_name = "ensemble"

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        weights: list[float] | None = None,
        rrf_k: int = _DEFAULT_RRF_K,
        candidate_multiplier: int = 3,
    ) -> None:
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must have the same length as retrievers")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        fetch_k = top_k * self.candidate_multiplier
        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            results = retriever.retrieve(query, top_k=fetch_k, filters=filters)
            for rank, chunk in enumerate(results, start=1):
                fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + weight / (
                    self.rrf_k + rank
                )
                chunk_lookup.setdefault(chunk.chunk_id, chunk)

        ranked_ids = sorted(fused_scores, key=lambda cid: -fused_scores[cid])[:top_k]
        return [
            chunk_lookup[cid].model_copy(
                update={"score": fused_scores[cid], "source_retriever": self.retriever_name}
            )
            for cid in ranked_ids
        ]
