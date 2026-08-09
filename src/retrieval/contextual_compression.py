"""Contextual compression retrieval: shrink each retrieved chunk down to the
sentences actually relevant to the query, instead of handing the full chunk
(possibly mixing relevant and irrelevant sentences) to the context window.

The standard implementation of this pattern (LangChain's
ContextualCompressionRetriever) uses an LLM to abstractively compress each
chunk. No LLM abstraction exists yet (Milestone 6-7), so `default_compressor`
below is a deterministic extractive heuristic — keep sentences that share a
content word with the query, drop the rest — real and useful today, with a
documented upgrade path once an LLM is available for abstractive compression.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.chunking.text_utils import split_sentences
from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk
from src.utils.text import content_words

CompressionFn = Callable[[str, str], str]


def default_compressor(query: str, text: str) -> str:
    query_words = content_words(query)
    if not query_words:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text

    kept = [s for s in sentences if content_words(s) & query_words]
    # Never compress a chunk down to nothing — an imperfect original beats an empty result.
    return " ".join(kept) if kept else text


class ContextualCompressionRetriever(BaseRetriever):
    retriever_name = "contextual_compression"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        compression_fn: CompressionFn = default_compressor,
    ) -> None:
        self.base_retriever = base_retriever
        self.compression_fn = compression_fn

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        results = self.base_retriever.retrieve(query, top_k=top_k, filters=filters)
        compressed = []
        for r in results:
            new_text = self.compression_fn(query, r.text)
            compressed.append(
                r.model_copy(update={"text": new_text, "source_retriever": self.retriever_name})
            )
        return compressed
