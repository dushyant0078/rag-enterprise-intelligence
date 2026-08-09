from __future__ import annotations

from src.retrieval.multi_query import MultiQueryRetriever, default_expansion
from tests.retrieval.conftest import FakeRetriever, make_chunk


def test_default_expansion_strips_stopwords() -> None:
    variants = default_expansion("what is the notice period for termination")
    assert any("notice" in v and "period" in v and "termination" in v for v in variants)
    assert all("what" not in v for v in variants)


def test_default_expansion_splits_compound_and_queries() -> None:
    variants = default_expansion("parental leave and sick leave policy")
    assert any("parental leave" in v for v in variants)
    assert any("sick leave policy" in v for v in variants)


def test_default_expansion_no_change_needed_returns_no_variants() -> None:
    # Every word is already a content word, no "and" to split -- nothing to add.
    variants = default_expansion("parental leave duration")
    assert variants == []


def test_multi_query_merges_results_across_variants() -> None:
    base = FakeRetriever([make_chunk("a", 0.9), make_chunk("b", 0.3)])
    retriever = MultiQueryRetriever(base, expansion_fn=lambda q: ["variant one", "variant two"])

    results = retriever.retrieve("original query", top_k=5)
    # base retriever returns the same fixed list regardless of query text here, so merging
    # should dedupe by chunk_id and keep the highest score seen for each.
    assert {r.chunk_id for r in results} == {"a", "b"}
    assert len(base.calls) == 3  # original + 2 variants


def test_multi_query_keeps_best_score_per_chunk() -> None:
    class VariantAwareRetriever(FakeRetriever):
        def retrieve(self, query, top_k=10, filters=None):
            self.calls.append((query, top_k, filters))
            if query == "low":
                return [make_chunk("a", 0.2)]
            return [make_chunk("a", 0.9)]

    base = VariantAwareRetriever([])
    retriever = MultiQueryRetriever(base, expansion_fn=lambda q: ["low"])

    results = retriever.retrieve("high", top_k=5)
    assert results[0].score == 0.9  # kept the better of the two scores for "a"


def test_multi_query_respects_top_k() -> None:
    base = FakeRetriever([make_chunk(str(i), float(i)) for i in range(10)])
    retriever = MultiQueryRetriever(base, expansion_fn=lambda q: [])
    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 3
