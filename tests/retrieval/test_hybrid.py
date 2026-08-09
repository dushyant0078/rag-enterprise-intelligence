from __future__ import annotations

from src.retrieval.hybrid import HybridRetriever
from tests.retrieval.conftest import FakeRetriever, make_chunk


def test_hybrid_favors_chunk_strong_in_both(make_retrieved_chunk) -> None:
    dense = FakeRetriever([make_chunk("a", 0.9), make_chunk("b", 0.5), make_chunk("c", 0.1)])
    sparse = FakeRetriever([make_chunk("b", 10.0), make_chunk("a", 5.0), make_chunk("c", 1.0)])
    retriever = HybridRetriever(dense, sparse, dense_weight=0.5, sparse_weight=0.5)

    results = retriever.retrieve("query", top_k=3)
    ids = [r.chunk_id for r in results]
    # "a" and "b" both score reasonably on both signals; "c" is weak on both -- should rank last.
    assert ids[-1] == "c"
    assert set(ids[:2]) == {"a", "b"}
    assert all(r.source_retriever == "hybrid" for r in results)


def test_hybrid_dense_only_weighting() -> None:
    dense = FakeRetriever([make_chunk("a", 0.9), make_chunk("b", 0.1)])
    sparse = FakeRetriever([make_chunk("b", 100.0), make_chunk("a", 1.0)])
    retriever = HybridRetriever(dense, sparse, dense_weight=1.0, sparse_weight=0.0)

    results = retriever.retrieve("query", top_k=2)
    assert results[0].chunk_id == "a"  # sparse signal fully ignored


def test_hybrid_includes_chunk_found_by_only_one_retriever() -> None:
    dense = FakeRetriever([make_chunk("a", 0.9)])
    sparse = FakeRetriever([make_chunk("b", 5.0)])
    retriever = HybridRetriever(dense, sparse, dense_weight=0.5, sparse_weight=0.5)

    results = retriever.retrieve("query", top_k=5)
    assert {r.chunk_id for r in results} == {"a", "b"}


def test_hybrid_respects_top_k() -> None:
    dense = FakeRetriever([make_chunk(str(i), float(i)) for i in range(10)])
    sparse = FakeRetriever([])
    retriever = HybridRetriever(dense, sparse)
    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 3


def test_hybrid_passes_filters_through_to_both_retrievers() -> None:
    dense = FakeRetriever([make_chunk("a", 0.9, domain="hr"), make_chunk("b", 0.5, domain="legal")])
    sparse = FakeRetriever(
        [make_chunk("a", 5.0, domain="hr"), make_chunk("b", 3.0, domain="legal")]
    )
    retriever = HybridRetriever(dense, sparse)

    retriever.retrieve("query", top_k=5, filters={"domain": "hr"})
    assert dense.calls[0][2] == {"domain": "hr"}
    assert sparse.calls[0][2] == {"domain": "hr"}
