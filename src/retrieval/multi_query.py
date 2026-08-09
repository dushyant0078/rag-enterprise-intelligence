"""Multi-query retrieval: run several phrasings of the same query and merge
results, catching relevant chunks that a single phrasing's embedding/lexical
match would miss.

The standard implementation of this pattern (LangChain's MultiQueryRetriever,
etc.) uses an LLM to generate the alternate phrasings. No LLM abstraction
exists yet (that's Milestone 6-7), so `default_expansion` below is a
deterministic heuristic stand-in — strip a query down to its content words,
and split an "X and Y" compound question into two — good enough to
demonstrate the retrieval-merging mechanics for real, but a real quality
upgrade once `src/llm` exists. Callers can already pass their own
`expansion_fn` today (e.g. an LLM-backed one) without any interface change.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk
from src.utils.text import STOPWORDS, tokenize

ExpansionFn = Callable[[str], list[str]]


def default_expansion(query: str) -> list[str]:
    variants: list[str] = []

    words = tokenize(query)
    keywords = [w for w in words if w not in STOPWORDS]
    if keywords and len(keywords) < len(words):
        variants.append(" ".join(keywords))

    if " and " in query.lower():
        parts = re.split(r"\band\b", query, flags=re.IGNORECASE)
        variants.extend(p.strip() for p in parts if p.strip())

    return variants


class MultiQueryRetriever(BaseRetriever):
    retriever_name = "multi_query"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        expansion_fn: ExpansionFn = default_expansion,
    ) -> None:
        self.base_retriever = base_retriever
        self.expansion_fn = expansion_fn

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        variants = [query] + [v for v in self.expansion_fn(query) if v and v != query]

        best_by_id: dict[str, RetrievedChunk] = {}
        for variant in variants:
            for chunk in self.base_retriever.retrieve(variant, top_k=top_k, filters=filters):
                existing = best_by_id.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    best_by_id[chunk.chunk_id] = chunk

        ranked = sorted(best_by_id.values(), key=lambda c: -c.score)[:top_k]
        return [c.model_copy(update={"source_retriever": self.retriever_name}) for c in ranked]
