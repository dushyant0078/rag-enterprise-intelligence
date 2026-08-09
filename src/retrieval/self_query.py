"""Self-query retrieval: pull structured filters (domain, table-only) out of
a natural-language query and pass them through to the underlying retriever
as real metadata filters, instead of leaving that information to be matched
lexically/semantically.

The standard implementation of this pattern (LangChain's SelfQueryRetriever)
uses an LLM to parse arbitrary structured filters from free text. No LLM
abstraction exists yet (Milestone 6-7), so this is a regex-based heuristic
covering a defined, narrow set of patterns — genuinely functional for those
patterns, not a simulation of the general capability. Documented as a
deliberately scoped-down stand-in, not passed off as the full technique.
"""

from __future__ import annotations

import re
from typing import Any

from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk

_DOMAIN_PATTERN = re.compile(
    r"\b(?:in|from|within)\s+the\s+(\w+)\s+domain\b"
    r"|\b(?:in|from|within)\s+(\w+)\s+documents?\b",
    re.IGNORECASE,
)
_TABLE_ONLY_PATTERN = re.compile(r"\b(?:in|from)\s+a\s+table\b", re.IGNORECASE)


def parse_structured_filters(query: str, known_domains: list[str]) -> tuple[str, dict[str, Any]]:
    """Returns (cleaned_query, filters). Matched filter phrases are stripped
    from the query text passed on to the underlying retriever, so they don't
    also (redundantly, and sometimes misleadingly) participate in the
    similarity/lexical match itself.
    """
    filters: dict[str, Any] = {}
    cleaned = query

    domain_match = _DOMAIN_PATTERN.search(query)
    if domain_match:
        candidate = (domain_match.group(1) or domain_match.group(2)).lower()
        if candidate in known_domains:
            filters["domain"] = candidate
            cleaned = cleaned[: domain_match.start()] + cleaned[domain_match.end() :]

    table_match = _TABLE_ONLY_PATTERN.search(cleaned)
    if table_match:
        filters["is_table"] = True
        cleaned = cleaned[: table_match.start()] + cleaned[table_match.end() :]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return (cleaned or query), filters


class SelfQueryRetriever(BaseRetriever):
    retriever_name = "self_query"

    def __init__(self, base_retriever: BaseRetriever, known_domains: list[str]) -> None:
        self.base_retriever = base_retriever
        self.known_domains = [d.lower() for d in known_domains]

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        cleaned_query, parsed_filters = parse_structured_filters(query, self.known_domains)
        merged_filters = {**(filters or {}), **parsed_filters}  # explicit filters win on conflict
        results = self.base_retriever.retrieve(cleaned_query, top_k=top_k, filters=merged_filters)
        return [r.model_copy(update={"source_retriever": self.retriever_name}) for r in results]
