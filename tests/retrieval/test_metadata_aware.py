from __future__ import annotations

from src.retrieval.metadata_aware import MetadataAwareRetriever
from tests.retrieval.conftest import FakeRetriever, make_chunk


def test_unrestricted_when_allowed_domains_is_none() -> None:
    base = FakeRetriever([make_chunk("a", 0.9, domain="hr"), make_chunk("b", 0.8, domain="legal")])
    retriever = MetadataAwareRetriever(base, allowed_domains=None)
    results = retriever.retrieve("query", top_k=5)
    assert {r.chunk_id for r in results} == {"a", "b"}


def test_restricts_to_allowed_domains() -> None:
    base = FakeRetriever([make_chunk("a", 0.9, domain="hr"), make_chunk("b", 0.8, domain="legal")])
    retriever = MetadataAwareRetriever(base, allowed_domains=["hr"])
    results = retriever.retrieve("query", top_k=5)
    assert base.calls[0][2] == {"domain": ["hr"]}
    assert {r.chunk_id for r in results} == {"a"}


def test_empty_allowed_domains_blocks_everything() -> None:
    base = FakeRetriever([make_chunk("a", 0.9, domain="hr")])
    retriever = MetadataAwareRetriever(base, allowed_domains=[])
    results = retriever.retrieve("query", top_k=5)
    assert results == []
    assert base.calls == []  # short-circuited before even calling the base retriever


def test_caller_filter_can_only_narrow_not_escape_allow_list() -> None:
    base = FakeRetriever([])
    retriever = MetadataAwareRetriever(base, allowed_domains=["hr", "legal"])
    retriever.retrieve("query", top_k=5, filters={"domain": ["legal", "banking"]})
    # "banking" isn't in the allow-list, so it's dropped -- only "legal" remains.
    assert base.calls[0][2] == {"domain": ["legal"]}


def test_caller_filter_single_value_intersected() -> None:
    base = FakeRetriever([])
    retriever = MetadataAwareRetriever(base, allowed_domains=["hr"])
    results = retriever.retrieve("query", top_k=5, filters={"domain": "legal"})
    # "legal" isn't in the allow-list, so the intersection is empty -- short-circuited to
    # no results without even calling the base retriever, same as the empty-allow-list case.
    assert results == []
    assert base.calls == []


def test_source_retriever_tagged() -> None:
    base = FakeRetriever([make_chunk("a", 0.9, domain="hr")])
    retriever = MetadataAwareRetriever(base, allowed_domains=["hr"])
    results = retriever.retrieve("query", top_k=5)
    assert results[0].source_retriever == "metadata_aware"
