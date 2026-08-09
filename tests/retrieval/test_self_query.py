from __future__ import annotations

from src.retrieval.self_query import SelfQueryRetriever, parse_structured_filters
from tests.retrieval.conftest import FakeRetriever, make_chunk

_KNOWN_DOMAINS = ["hr", "legal", "banking", "insurance"]


def test_parse_domain_filter_in_the_x_domain() -> None:
    cleaned, filters = parse_structured_filters(
        "what is the notice period in the legal domain", _KNOWN_DOMAINS
    )
    assert filters == {"domain": "legal"}
    assert "legal" not in cleaned or "domain" not in cleaned


def test_parse_domain_filter_from_x_documents() -> None:
    cleaned, filters = parse_structured_filters(
        "show me policies from hr documents", _KNOWN_DOMAINS
    )
    assert filters == {"domain": "hr"}


def test_parse_ignores_unknown_domain() -> None:
    cleaned, filters = parse_structured_filters("find info in the marketing domain", _KNOWN_DOMAINS)
    assert filters == {}
    assert cleaned == "find info in the marketing domain"


def test_parse_table_only_filter() -> None:
    cleaned, filters = parse_structured_filters(
        "what deductibles are listed in a table", _KNOWN_DOMAINS
    )
    assert filters == {"is_table": True}


def test_parse_no_filters_returns_original_query() -> None:
    cleaned, filters = parse_structured_filters("what is the deductible amount", _KNOWN_DOMAINS)
    assert filters == {}
    assert cleaned == "what is the deductible amount"


def test_self_query_retriever_passes_parsed_filters_through() -> None:
    base = FakeRetriever([make_chunk("a", 0.9, domain="legal")])
    retriever = SelfQueryRetriever(base, known_domains=_KNOWN_DOMAINS)

    results = retriever.retrieve("termination notice in the legal domain", top_k=5)
    assert base.calls[0][2] == {"domain": "legal"}
    assert results[0].source_retriever == "self_query"


def test_self_query_explicit_filters_merge_with_parsed() -> None:
    base = FakeRetriever([])
    retriever = SelfQueryRetriever(base, known_domains=_KNOWN_DOMAINS)

    retriever.retrieve("info in the hr domain", top_k=5, filters={"is_table": True})
    assert base.calls[0][2] == {"is_table": True, "domain": "hr"}
