from __future__ import annotations

import pytest

from src.retrieval.registry import available_strategies, describe


def test_available_strategies_lists_all_ten() -> None:
    strategies = available_strategies()
    expected = {
        "dense", "sparse_bm25", "hybrid", "multi_query", "self_query",
        "parent_child", "contextual_compression", "ensemble", "time_aware",
        "metadata_aware",
    }  # fmt: skip
    assert set(strategies) == expected


def test_describe_known_strategy() -> None:
    assert "DenseRetriever" in describe("dense")


def test_describe_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        describe("not_a_real_strategy")
