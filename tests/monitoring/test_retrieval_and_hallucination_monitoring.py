from __future__ import annotations

from src.monitoring.hallucination_alerts import HallucinationAlertMonitor
from src.monitoring.retrieval_monitoring import RetrievalMonitor
from src.retrieval.schemas import RetrievedChunk


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", doc_id="d1", text="text", score=score)


def test_retrieval_monitor_tracks_empty_rate() -> None:
    mon = RetrievalMonitor()
    mon.record("q1", [_chunk(0.9)])
    mon.record("q2", [])
    mon.record("q3", [])
    assert mon.empty_result_rate() == 2 / 3


def test_retrieval_monitor_mean_top_score() -> None:
    mon = RetrievalMonitor()
    mon.record("q1", [_chunk(0.8), _chunk(0.5)])
    mon.record("q2", [_chunk(0.6)])
    assert abs(mon.mean_top_score() - 0.7) < 1e-9  # top scores are 0.8 and 0.6


def test_retrieval_monitor_respects_window_size() -> None:
    mon = RetrievalMonitor(window_size=3)
    for i in range(5):
        mon.record(f"q{i}", [_chunk(0.5)])
    assert mon.observation_count() == 3


def test_retrieval_monitor_empty_when_no_observations() -> None:
    mon = RetrievalMonitor()
    assert mon.empty_result_rate() == 0.0
    assert mon.mean_top_score() is None


def test_hallucination_monitor_no_alert_below_min_window() -> None:
    mon = HallucinationAlertMonitor(min_window=10)
    alert = None
    for _ in range(5):
        alert = mon.record(False)  # all ungrounded, but below min_window
    assert alert is None


def test_hallucination_monitor_alerts_above_threshold() -> None:
    mon = HallucinationAlertMonitor(min_window=10, ungrounded_rate_threshold=0.3)
    alert = None
    for i in range(10):
        alert = mon.record(i >= 4)  # 4/10 = 40% ungrounded
    assert alert is not None
    assert alert.name == "high_ungrounded_rate"
    assert abs(alert.context["ungrounded_rate"] - 0.4) < 1e-9


def test_hallucination_monitor_no_alert_when_mostly_grounded() -> None:
    mon = HallucinationAlertMonitor(min_window=10, ungrounded_rate_threshold=0.3)
    alert = None
    for i in range(10):
        alert = mon.record(i != 0)  # only 1/10 ungrounded
    assert alert is None


def test_hallucination_monitor_severity_escalates_at_double_threshold() -> None:
    mon = HallucinationAlertMonitor(min_window=10, ungrounded_rate_threshold=0.3)
    alert = None
    for _ in range(10):
        alert = mon.record(False)  # 100% ungrounded, well past 2x threshold
    assert alert.severity == "critical"


def test_hallucination_monitor_window_rolls_off_old_values() -> None:
    mon = HallucinationAlertMonitor(window_size=5, min_window=5, ungrounded_rate_threshold=0.5)
    for _ in range(5):
        mon.record(False)  # 100% ungrounded initially
    assert mon.ungrounded_rate() == 1.0
    for _ in range(5):
        mon.record(True)  # push all the ungrounded ones out of the window
    assert mon.ungrounded_rate() == 0.0
