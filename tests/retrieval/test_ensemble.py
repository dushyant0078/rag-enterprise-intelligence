from __future__ import annotations

from src.retrieval.ensemble import EnsembleRetriever
from tests.retrieval.conftest import FakeRetriever, make_chunk


def test_ensemble_ranks_by_reciprocal_rank_fusion() -> None:
    r1 = FakeRetriever([make_chunk("a", 1.0), make_chunk("b", 0.9), make_chunk("c", 0.8)])
    r2 = FakeRetriever([make_chunk("b", 5.0), make_chunk("a", 4.0), make_chunk("c", 3.0)])
    ensemble = EnsembleRetriever([r1, r2])

    results = ensemble.retrieve("query", top_k=3)
    ids = [r.chunk_id for r in results]
    # "a" is rank 1 in r1 / rank 2 in r2; "b" is rank 2 in r1 / rank 1 in r2 -- roughly
    # symmetric, both should outrank "c" which is rank 3 in both.
    assert ids[-1] == "c"
    assert all(r.source_retriever == "ensemble" for r in results)


def test_ensemble_weights_change_ranking() -> None:
    r1 = FakeRetriever([make_chunk("a", 1.0), make_chunk("b", 0.5)])
    r2 = FakeRetriever([make_chunk("b", 1.0), make_chunk("a", 0.5)])

    equal_weight = EnsembleRetriever([r1, r2], weights=[1.0, 1.0])
    results_equal = equal_weight.retrieve("query", top_k=2)
    assert results_equal[0].score == results_equal[1].score  # perfectly symmetric

    r1_favored = EnsembleRetriever([r1, r2], weights=[10.0, 1.0])
    results_favored = r1_favored.retrieve("query", top_k=2)
    assert results_favored[0].chunk_id == "a"  # r1 ranks "a" first and now dominates


def test_ensemble_mismatched_weights_length_raises() -> None:
    import pytest

    r1 = FakeRetriever([])
    r2 = FakeRetriever([])
    with pytest.raises(ValueError):
        EnsembleRetriever([r1, r2], weights=[1.0])


def test_ensemble_single_retriever_preserves_order() -> None:
    r1 = FakeRetriever([make_chunk("a", 1.0), make_chunk("b", 0.5), make_chunk("c", 0.1)])
    ensemble = EnsembleRetriever([r1])
    results = ensemble.retrieve("query", top_k=3)
    assert [r.chunk_id for r in results] == ["a", "b", "c"]


def test_ensemble_respects_top_k() -> None:
    r1 = FakeRetriever([make_chunk(str(i), float(10 - i)) for i in range(10)])
    ensemble = EnsembleRetriever([r1])
    results = ensemble.retrieve("query", top_k=4)
    assert len(results) == 4
