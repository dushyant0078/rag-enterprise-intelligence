"""Hallucination alerting: tracks a rolling grounded/ungrounded rate over
recent responses and raises a real Alert when it crosses a configurable
threshold. Distinct from Milestone 8's benchmark-based hallucination rate
metric -- this watches live traffic continuously, not a fixed labeled set
run on demand.
"""

from __future__ import annotations

from src.monitoring.schemas import Alert

DEFAULT_MIN_WINDOW = 10  # don't alert on noise from the first few requests
DEFAULT_UNGROUNDED_RATE_THRESHOLD = 0.3  # alert once 30%+ of recent responses are ungrounded


class HallucinationAlertMonitor:
    def __init__(
        self,
        window_size: int = 50,
        ungrounded_rate_threshold: float = DEFAULT_UNGROUNDED_RATE_THRESHOLD,
        min_window: int = DEFAULT_MIN_WINDOW,
    ) -> None:
        self.window_size = window_size
        self.ungrounded_rate_threshold = ungrounded_rate_threshold
        self.min_window = min_window
        self._is_grounded_flags: list[bool] = []

    def record(self, is_grounded: bool) -> Alert | None:
        self._is_grounded_flags.append(is_grounded)
        if len(self._is_grounded_flags) > self.window_size:
            self._is_grounded_flags.pop(0)

        if len(self._is_grounded_flags) < self.min_window:
            return None

        rate = self.ungrounded_rate()
        if rate >= self.ungrounded_rate_threshold:
            return Alert(
                name="high_ungrounded_rate",
                severity="critical" if rate >= self.ungrounded_rate_threshold * 2 else "warning",
                message=(
                    f"{rate:.0%} of the last {len(self._is_grounded_flags)} responses were "
                    f"ungrounded (threshold: {self.ungrounded_rate_threshold:.0%})"
                ),
                context={"ungrounded_rate": rate, "window_size": len(self._is_grounded_flags)},
            )
        return None

    def ungrounded_rate(self) -> float:
        if not self._is_grounded_flags:
            return 0.0
        ungrounded = sum(1 for g in self._is_grounded_flags if not g)
        return ungrounded / len(self._is_grounded_flags)

    def grounded_rate(self) -> float:
        return 1.0 - self.ungrounded_rate()
