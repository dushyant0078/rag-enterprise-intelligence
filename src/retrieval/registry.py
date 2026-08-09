"""Retrieval strategy registry. Unlike the parsing/chunking registries, most
retrieval strategies wrap another retriever rather than being constructible
from scratch with defaults alone — this module documents the dependency each
name needs rather than pretending a uniform `get_retriever(name)` call would
make sense for all ten.
"""

from __future__ import annotations

AVAILABLE_STRATEGIES: dict[str, str] = {
    "dense": "DenseRetriever(embedding_model, vector_store, namespace=...)",
    "sparse_bm25": "BM25Retriever(records)",
    "hybrid": "HybridRetriever(dense_retriever, sparse_retriever, dense_weight=, sparse_weight=)",
    "multi_query": "MultiQueryRetriever(base_retriever, expansion_fn=...)",
    "self_query": "SelfQueryRetriever(base_retriever, known_domains)",
    "parent_child": "ParentChildRetriever(child_retriever, parent_lookup)",
    "contextual_compression": "ContextualCompressionRetriever(base_retriever, compression_fn=...)",
    "ensemble": "EnsembleRetriever(retrievers, weights=...)",
    "time_aware": "TimeAwareRetriever(base_retriever, half_life_days=...)",
    "metadata_aware": "MetadataAwareRetriever(base_retriever, allowed_domains)",
}


def available_strategies() -> tuple[str, ...]:
    return tuple(AVAILABLE_STRATEGIES.keys())


def describe(strategy: str) -> str:
    """Return the constructor signature a caller needs for this strategy —
    a lookup aid, since these can't be built from a name + kwargs alone.
    """
    if strategy not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"Unknown retrieval strategy '{strategy}'. Available: {available_strategies()}"
        )
    return AVAILABLE_STRATEGIES[strategy]
