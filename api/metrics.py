"""In-memory request metrics for /metrics. Real Prometheus text exposition
format (https://prometheus.io/docs/instrumenting/exposition_formats/), so
this is scrapeable by an actual Prometheus instance today -- not a mock. The
full OpenTelemetry tracing/collector pipeline from docs/02_architecture.md
is Milestone 11's job; this is the request-counting layer that would feed it.
"""

from __future__ import annotations

import threading


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_counts: dict[tuple[str, str, int], int] = (
            {}
        )  # (method, path, status) -> count
        self._latency_sum_seconds: dict[tuple[str, str], float] = {}
        self._latency_count: dict[tuple[str, str], int] = {}

    def record(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        with self._lock:
            key = (method, path, status_code)
            self._request_counts[key] = self._request_counts.get(key, 0) + 1

            latency_key = (method, path)
            self._latency_sum_seconds[latency_key] = (
                self._latency_sum_seconds.get(latency_key, 0.0) + duration_seconds
            )
            self._latency_count[latency_key] = self._latency_count.get(latency_key, 0) + 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP rag_api_requests_total Total HTTP requests processed.",
                "# TYPE rag_api_requests_total counter",
            ]
            for (method, path, status), count in sorted(self._request_counts.items()):
                labels = f'method="{method}",path="{path}",status="{status}"'
                lines.append(f"rag_api_requests_total{{{labels}}} {count}")

            lines += [
                "# HELP rag_api_request_latency_seconds_sum Sum of request latencies (s).",
                "# TYPE rag_api_request_latency_seconds_sum counter",
            ]
            for (method, path), total in sorted(self._latency_sum_seconds.items()):
                labels = f'method="{method}",path="{path}"'
                lines.append(f"rag_api_request_latency_seconds_sum{{{labels}}} {total:.6f}")

            lines += [
                "# HELP rag_api_request_latency_seconds_count Count of requests measured.",
                "# TYPE rag_api_request_latency_seconds_count counter",
            ]
            for (method, path), count in sorted(self._latency_count.items()):
                labels = f'method="{method}",path="{path}"'
                lines.append(f"rag_api_request_latency_seconds_count{{{labels}}} {count}")

            return "\n".join(lines) + "\n"
