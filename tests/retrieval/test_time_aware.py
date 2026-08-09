from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.retrieval.time_aware import TimeAwareRetriever
from tests.retrieval.conftest import FakeRetriever, make_chunk


def test_older_chunk_loses_to_newer_chunk_with_similar_score() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_date = (now - timedelta(days=365)).isoformat()
    new_date = (now - timedelta(days=1)).isoformat()

    base = FakeRetriever(
        [
            make_chunk("old", 0.95, metadata={"effective_date": old_date}),
            make_chunk("new", 0.90, metadata={"effective_date": new_date}),
        ]
    )
    retriever = TimeAwareRetriever(base, half_life_days=180, now=now)
    results = retriever.retrieve("query", top_k=2)

    assert results[0].chunk_id == "new"  # recency flipped the ranking despite the lower raw score


def test_chunk_with_no_date_metadata_is_not_penalized() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    base = FakeRetriever([make_chunk("undated", 0.5, metadata={})])
    retriever = TimeAwareRetriever(base, half_life_days=180, now=now)
    results = retriever.retrieve("query", top_k=1)
    assert results[0].score == 0.5  # weight of 1.0 applied -- no penalty


def test_exactly_one_half_life_old_halves_the_score() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    half_life_ago = (now - timedelta(days=180)).isoformat()
    base = FakeRetriever([make_chunk("a", 1.0, metadata={"effective_date": half_life_ago})])
    retriever = TimeAwareRetriever(base, half_life_days=180, now=now)
    results = retriever.retrieve("query", top_k=1)
    assert abs(results[0].score - 0.5) < 0.01


def test_falls_back_to_ingested_at_when_effective_date_missing() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_date = (now - timedelta(days=365)).isoformat()
    base = FakeRetriever([make_chunk("a", 1.0, metadata={"ingested_at": old_date})])
    retriever = TimeAwareRetriever(base, half_life_days=180, now=now)
    results = retriever.retrieve("query", top_k=1)
    assert results[0].score < 1.0  # date was found and applied


def test_source_retriever_tagged() -> None:
    base = FakeRetriever([make_chunk("a", 1.0)])
    retriever = TimeAwareRetriever(base)
    results = retriever.retrieve("query", top_k=1)
    assert results[0].source_retriever == "time_aware"
