"""Base reranker interface. Every backend in src/reranking/ implements this —
reranking takes the top-N candidates a retriever already found and rescores
them with a (usually more expensive, more accurate) model that can look at
the query and each candidate together, rather than comparing independently
computed embeddings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.schemas import RetrievedChunk


class BaseReranker(ABC):
    reranker_name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Rescore and reorder `chunks` for `query`. Returns at most top_k
        chunks (or all of them, reordered, if top_k is None).
        """
        raise NotImplementedError

    def _annotate(
        self, chunk: RetrievedChunk, new_score: float, original_rank: int
    ) -> RetrievedChunk:
        """Shared bookkeeping every reranker uses: preserve the pre-rerank
        score/rank in metadata (what the retrieval-vs-reranked comparison in
        docs/06_reranking_and_prompting.md reads), and tag provenance so a
        downstream consumer can see this chunk passed through a reranker.
        """
        metadata = dict(chunk.metadata)
        metadata["pre_rerank_score"] = chunk.score
        metadata["pre_rerank_rank"] = original_rank
        return chunk.model_copy(
            update={
                "score": new_score,
                "metadata": metadata,
                "source_retriever": f"{chunk.source_retriever}+{self.reranker_name}",
            }
        )
