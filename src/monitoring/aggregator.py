"""MonitoringAggregator: the single object the API layer records every
pipeline response into, and reads a monitoring snapshot back out of --
ties together token/cost tracking, retrieval monitoring, hallucination
alerts, and feedback into one place, the way api/state.AppState ties
together the pipeline's own components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.citations.schemas import ValidatedAnswer
from src.monitoring.cost_tracking import CostCalculator
from src.monitoring.feedback import FeedbackStore
from src.monitoring.hallucination_alerts import HallucinationAlertMonitor
from src.monitoring.retrieval_monitoring import RetrievalMonitor
from src.monitoring.schemas import Alert
from src.monitoring.token_tracking import TokenUsageTracker


class MonitoringAggregator:
    def __init__(
        self,
        feedback_dir: Path,
        cost_calculator: CostCalculator | None = None,
        hallucination_threshold: float = 0.3,
    ) -> None:
        self.token_tracker = TokenUsageTracker()
        self.cost_calculator = cost_calculator or CostCalculator()
        self.retrieval_monitor = RetrievalMonitor()
        self.hallucination_monitor = HallucinationAlertMonitor(
            ungrounded_rate_threshold=hallucination_threshold
        )
        self.feedback_store = FeedbackStore(feedback_dir)
        self.recent_alerts: list[Alert] = []

    def record_response(self, query: str, result: ValidatedAnswer) -> Alert | None:
        """Call once per pipeline response (see api/routers/chat.py and
        query.py) -- updates every tracker from one ValidatedAnswer's
        metadata and retrieved passages, and returns a hallucination Alert
        if this response pushed the rolling ungrounded rate over threshold.
        """
        metadata = result.metadata or {}
        model = metadata.get("model")
        if model:
            self.token_tracker.record(
                model, metadata.get("input_tokens"), metadata.get("output_tokens")
            )

        self.retrieval_monitor.record(query, result.retrieved_passages)

        alert = self.hallucination_monitor.record(result.is_grounded)
        if alert:
            self.recent_alerts.append(alert)
            self.recent_alerts = self.recent_alerts[
                -20:
            ]  # bounded, most-recent-first isn't needed here
        return alert

    def snapshot(self) -> dict[str, Any]:
        usage_by_model = self.token_tracker.total_by_model()
        return {
            "token_usage": {
                "total_requests": self.token_tracker.total_requests(),
                "total_tokens": self.token_tracker.total_tokens(),
                "by_model": usage_by_model,
            },
            "estimated_cost_usd": self.cost_calculator.estimate_total(usage_by_model),
            "retrieval": {
                "observations": self.retrieval_monitor.observation_count(),
                "empty_result_rate": self.retrieval_monitor.empty_result_rate(),
                "mean_top_score": self.retrieval_monitor.mean_top_score(),
            },
            "hallucination": {
                "ungrounded_rate": self.hallucination_monitor.ungrounded_rate(),
                "grounded_rate": self.hallucination_monitor.grounded_rate(),
                "recent_alerts": [a.model_dump(mode="json") for a in self.recent_alerts],
            },
            "feedback": self.feedback_store.summary(),
        }
