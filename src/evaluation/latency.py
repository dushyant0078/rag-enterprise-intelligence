"""Response latency aggregation. Real, exact statistics over timing data
already collected by LLMResponse.latency_ms (Milestone 7) -- no proxy needed
here, latency is directly measurable.
"""

from __future__ import annotations

import statistics


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 100]). Simple and deterministic --
    exact interpolation methods vary and aren't worth the complexity at the
    benchmark sizes this project runs.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1)))))
    return ordered[index]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "mean": statistics.mean(latencies_ms),
    }
