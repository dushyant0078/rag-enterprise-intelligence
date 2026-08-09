"""Cohere rerank API backend.

Cohere's reranker is a hosted API — there's no local/offline mode. This
class is correct, complete code against the real `cohere` v7+ Python SDK; it
needs a Cohere API key and network access to Cohere's API (not in this
sandbox's allowlist) to actually run — same "correct, not verified here"
treatment as the Pinecone and Weaviate vector store backends in Milestone 5.
"""

from __future__ import annotations

import cohere

from src.reranking.base import BaseReranker
from src.retrieval.schemas import RetrievedChunk

DEFAULT_MODEL = "rerank-english-v3.0"


class CohereReranker(BaseReranker):
    reranker_name = "cohere_rerank"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = cohere.Client(api_key=api_key)
        self.model = model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=[c.text for c in chunks],
            top_n=top_k or len(chunks),
        )
        # Cohere returns results already sorted by relevance, each carrying the original
        # candidate's index -- no local re-sort needed, unlike the cross-encoder backend.
        return [
            self._annotate(chunks[result.index], float(result.relevance_score), result.index)
            for result in response.results
        ]
