from __future__ import annotations

import pytest

from src.reranking.lexical_overlap import LexicalOverlapReranker
from src.reranking.registry import UnknownRerankerError, available_rerankers, get_reranker


def test_available_rerankers_lists_all_four() -> None:
    rerankers = available_rerankers()
    for name in ("bge_reranker", "cross_encoder_ms_marco", "cohere_rerank", "lexical_overlap"):
        assert name in rerankers


def test_get_reranker_lexical_overlap() -> None:
    reranker = get_reranker("lexical_overlap")
    assert isinstance(reranker, LexicalOverlapReranker)


def test_get_reranker_unknown_raises() -> None:
    with pytest.raises(UnknownRerankerError):
        get_reranker("not_a_real_reranker")
