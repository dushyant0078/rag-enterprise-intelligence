"""Time-aware retrieval: reweight similarity scores by recency, so a highly
similar but outdated clause loses to a less textually similar but current
one (the recency-vs-relevance trade-off named explicitly in
docs/01_business_understanding.md's retrieval challenges).

Recency weight uses exponential decay: weight = 0.5 ** (age_days / half_life).
A chunk exactly one half-life old is worth half a same-age chunk's score; a
chunk with no usable date metadata gets weight 1.0 (no penalty) rather than
being silently deprioritized for a data-quality gap that isn't its fault.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk

_DATE_METADATA_KEYS = ("effective_date", "ingested_at")


def _parse_date(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _recency_weight(chunk: RetrievedChunk, now: datetime, half_life_days: float) -> float:
    for key in _DATE_METADATA_KEYS:
        parsed = _parse_date(chunk.metadata.get(key))
        if parsed is not None:
            age_days = max(0.0, (now - parsed).total_seconds() / 86400)
            return 0.5 ** (age_days / half_life_days)
    return 1.0  # no usable date — don't penalize for missing metadata


class TimeAwareRetriever(BaseRetriever):
    retriever_name = "time_aware"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        half_life_days: float = 180,
        candidate_multiplier: int = 3,
        now: datetime | None = None,
    ) -> None:
        self.base_retriever = base_retriever
        self.half_life_days = half_life_days
        self.candidate_multiplier = candidate_multiplier
        self._fixed_now = now  # override for deterministic tests

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        fetch_k = top_k * self.candidate_multiplier
        results = self.base_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        now = self._fixed_now or datetime.now(UTC)

        reweighted = [
            r.model_copy(
                update={
                    "score": r.score * _recency_weight(r, now, self.half_life_days),
                    "source_retriever": self.retriever_name,
                }
            )
            for r in results
        ]
        reweighted.sort(key=lambda r: -r.score)
        return reweighted[:top_k]
