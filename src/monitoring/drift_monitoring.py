"""Data drift monitoring: a real two-sample Kolmogorov-Smirnov test
(scipy.stats.ks_2samp -- the standard tool for exactly this comparison, not
a bespoke heuristic) comparing a stored baseline distribution of retrieval
scores against a current window. A statistically significant shift can mean
the corpus changed (new documents skew what's retrievable), the query mix
changed (a new class of question users are asking), or the embedding/
retrieval configuration changed -- this flags *that something shifted*, not
which of those it was.
"""

from __future__ import annotations

from scipy import stats

from src.monitoring.schemas import DriftReport

DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05
MIN_SAMPLE_SIZE = 10  # KS test results are unreliable below this; refuse rather than mislead


class InsufficientDataError(ValueError):
    pass


class DriftMonitor:
    def __init__(self, significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD) -> None:
        self.significance_threshold = significance_threshold
        self._baseline: list[float] | None = None

    def set_baseline(self, values: list[float]) -> None:
        if len(values) < MIN_SAMPLE_SIZE:
            raise InsufficientDataError(
                f"Baseline needs at least {MIN_SAMPLE_SIZE} values, got {len(values)}"
            )
        self._baseline = list(values)

    def has_baseline(self) -> bool:
        return self._baseline is not None

    def check(
        self, current_values: list[float], metric: str = "retrieval_top_score"
    ) -> DriftReport:
        if self._baseline is None:
            raise InsufficientDataError("No baseline set -- call set_baseline() first")
        if len(current_values) < MIN_SAMPLE_SIZE:
            raise InsufficientDataError(
                f"Current window needs at least {MIN_SAMPLE_SIZE} values, got {len(current_values)}"
            )

        result = stats.ks_2samp(self._baseline, current_values)
        drifted = bool(result.pvalue < self.significance_threshold)

        return DriftReport(
            metric=metric,
            baseline_size=len(self._baseline),
            current_size=len(current_values),
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            drifted=drifted,
            threshold=self.significance_threshold,
        )
