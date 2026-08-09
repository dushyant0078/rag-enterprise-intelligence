"""Live retrieval performance monitoring: tracks score distributions and
empty-result rate over actual production queries -- distinct from
Milestone 8's evaluation framework, which measures retrieval quality
against a labeled benchmark. This has no ground truth to check against; it
watches for operational symptoms (queries returning nothing, score
distributions drifting) that a labeled benchmark run wouldn't catch between
evaluation runs.
"""

from __future__ import annotations

import statistics

from src.monitoring.schemas import RetrievalObservation
from src.retrieval.schemas import RetrievedChunk


class RetrievalMonitor:
    def __init__(self, window_size: int = 500) -> None:
        self.window_size = window_size
        self._observations: list[RetrievalObservation] = []

    def record(self, query: str, results: list[RetrievedChunk]) -> RetrievalObservation:
        scores = [r.score for r in results]
        observation = RetrievalObservation(
            query=query,
            result_count=len(results),
            top_score=max(scores) if scores else None,
            mean_score=statistics.mean(scores) if scores else None,
        )
        self._observations.append(observation)
        if len(self._observations) > self.window_size:
            self._observations.pop(0)
        return observation

    def empty_result_rate(self) -> float:
        if not self._observations:
            return 0.0
        empty = sum(1 for o in self._observations if o.result_count == 0)
        return empty / len(self._observations)

    def mean_top_score(self) -> float | None:
        scores = [o.top_score for o in self._observations if o.top_score is not None]
        return statistics.mean(scores) if scores else None

    def recent_scores(self) -> list[float]:
        """All top-scores in the current window -- the raw input
        DriftMonitor compares against a stored baseline.
        """
        return [o.top_score for o in self._observations if o.top_score is not None]

    def observation_count(self) -> int:
        return len(self._observations)
