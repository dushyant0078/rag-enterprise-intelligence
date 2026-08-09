from __future__ import annotations

from pathlib import Path

from src.citations.schemas import ValidatedAnswer
from src.monitoring.aggregator import MonitoringAggregator
from src.retrieval.schemas import RetrievedChunk


def _result(is_grounded: bool, model: str = "claude-sonnet-4-5") -> ValidatedAnswer:
    return ValidatedAnswer(
        answer_text="answer",
        original_answer_text="answer",
        confidence_score=0.9 if is_grounded else 0.1,
        is_grounded=is_grounded,
        retrieved_passages=[RetrievedChunk(chunk_id="c1", doc_id="d1", text="t", score=0.8)],
        metadata={"model": model, "input_tokens": 100, "output_tokens": 20},
    )


def test_record_response_updates_token_and_retrieval_trackers(tmp_path: Path) -> None:
    agg = MonitoringAggregator(feedback_dir=tmp_path)
    agg.record_response("query", _result(is_grounded=True))

    snapshot = agg.snapshot()
    assert snapshot["token_usage"]["total_requests"] == 1
    assert snapshot["retrieval"]["observations"] == 1


def test_record_response_returns_alert_when_threshold_crossed(tmp_path: Path) -> None:
    agg = MonitoringAggregator(feedback_dir=tmp_path, hallucination_threshold=0.3)
    agg.hallucination_monitor.min_window = 5

    alert = None
    for i in range(5):
        alert = agg.record_response("q", _result(is_grounded=i >= 3))  # 3/5 = 60% ungrounded
    assert alert is not None
    assert alert.name == "high_ungrounded_rate"
    assert len(agg.recent_alerts) == 1


def test_snapshot_reflects_feedback(tmp_path: Path) -> None:
    agg = MonitoringAggregator(feedback_dir=tmp_path)
    agg.feedback_store.record(session_id="s1", user_id="u1", rating="up")
    snapshot = agg.snapshot()
    assert snapshot["feedback"]["total"] == 1
    assert snapshot["feedback"]["up"] == 1


def test_snapshot_includes_estimated_cost(tmp_path: Path) -> None:
    agg = MonitoringAggregator(feedback_dir=tmp_path)
    agg.record_response("q", _result(is_grounded=True))
    snapshot = agg.snapshot()
    assert "estimated_cost_usd" in snapshot
    assert snapshot["estimated_cost_usd"] >= 0.0


def test_record_response_skips_token_tracking_without_model_metadata(tmp_path: Path) -> None:
    agg = MonitoringAggregator(feedback_dir=tmp_path)
    result = ValidatedAnswer(
        answer_text="a",
        original_answer_text="a",
        confidence_score=0.5,
        is_grounded=True,
        retrieved_passages=[],
        metadata={},
    )
    agg.record_response("q", result)
    assert agg.token_tracker.total_requests() == 0
