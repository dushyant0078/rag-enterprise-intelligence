from __future__ import annotations

from pathlib import Path

import pytest

from src.monitoring.feedback import FeedbackStore


def test_record_and_retrieve(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    record = store.record(session_id="s1", user_id="u1", rating="up", query="q", answer_excerpt="a")
    assert record.rating == "up"
    assert len(store.all_records()) == 1


def test_rejects_invalid_rating(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    with pytest.raises(ValueError, match="rating must be"):
        store.record(session_id="s1", user_id="u1", rating="sideways")


def test_summary_empty(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    assert store.summary() == {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0.0}


def test_summary_computes_satisfaction_rate(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    store.record(session_id="s1", user_id="u1", rating="up")
    store.record(session_id="s1", user_id="u1", rating="up")
    store.record(session_id="s1", user_id="u1", rating="down")
    summary = store.summary()
    assert summary["total"] == 3
    assert summary["up"] == 2
    assert summary["down"] == 1
    assert summary["satisfaction_rate"] == pytest.approx(2 / 3, rel=1e-3)


def test_persists_across_new_store_instance(tmp_path: Path) -> None:
    FeedbackStore(tmp_path).record(session_id="s1", user_id="u1", rating="up")
    reloaded = FeedbackStore(tmp_path)
    assert len(reloaded.all_records()) == 1


def test_answer_excerpt_truncated_to_280_chars(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    long_answer = "x" * 500
    record = store.record(session_id="s1", user_id="u1", rating="up", answer_excerpt=long_answer)
    assert len(record.answer_excerpt) == 280
