"""Metadata-aware retrieval: enforce a domain allow-list at the retrieval
boundary, merging it with (and never widening) whatever filters the caller
already passed. This is the enforcement *point* the architecture calls for —
"Security is enforced at the retrieval boundary" and "RBAC and namespace
filtering happen in the Retriever, so no downstream component ever sees a
document the requesting user isn't entitled to" (docs/02_architecture.md
principles #4). The actual entitlement *resolution* (who is this user, what
are they allowed to see) is the Authentication Layer's job — Milestone 9 —
which will call into this wrapper with the resolved allow-list.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk


class MetadataAwareRetriever(BaseRetriever):
    retriever_name = "metadata_aware"

    def __init__(self, base_retriever: BaseRetriever, allowed_domains: list[str] | None) -> None:
        """`allowed_domains=None` means unrestricted (e.g. an admin role or a
        single-tenant deployment with no domain segmentation) — an *empty
        list* is different and means "allowed to see nothing," so an
        upstream caller must not conflate "no restriction" with "no access."
        """
        self.base_retriever = base_retriever
        self.allowed_domains = allowed_domains

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        merged_filters = dict(filters or {})

        if self.allowed_domains is not None:
            requested = merged_filters.get("domain")
            if requested is None:
                merged_filters["domain"] = list(self.allowed_domains)
            else:
                requested_list = requested if isinstance(requested, list) else [requested]
                # Intersect rather than override: a caller-supplied filter can only narrow
                # access further, never escape the allow-list.
                merged_filters["domain"] = [d for d in requested_list if d in self.allowed_domains]

        if merged_filters.get("domain") == []:
            return (
                []
            )  # intersection was empty — nothing this caller is both asking for and allowed to see

        results = self.base_retriever.retrieve(query, top_k=top_k, filters=merged_filters)
        return [r.model_copy(update={"source_retriever": self.retriever_name}) for r in results]
