from __future__ import annotations

from src.evaluation.retrieval_metrics import (
    context_precision,
    context_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_basic() -> None:
    retrieved = ["banking_a", "hr_a", "banking_b", "legal_a"]
    assert precision_at_k(retrieved, {"banking"}, 2) == 0.5
    assert precision_at_k(retrieved, {"banking"}, 4) == 0.5


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], {"banking"}, 5) == 0.0


def test_recall_at_k_finds_relevant_domain() -> None:
    retrieved = ["hr_a", "legal_a", "banking_a"]
    assert recall_at_k(retrieved, {"banking"}, 3) == 1.0
    assert recall_at_k(retrieved, {"banking"}, 1) == 0.0  # not in top 1


def test_recall_at_k_no_relevant_ids() -> None:
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_recall_at_k_multiple_relevant_domains_partial_found() -> None:
    retrieved = ["banking_a", "hr_a"]
    assert recall_at_k(retrieved, {"banking", "legal"}, 2) == 0.5  # found banking, not legal


def test_mrr_hit_at_rank_one() -> None:
    assert mrr(["banking_a", "hr_a"], {"banking"}) == 1.0


def test_mrr_hit_at_rank_three() -> None:
    assert abs(mrr(["hr_a", "legal_a", "banking_a"], {"banking"}) - (1 / 3)) < 1e-9


def test_mrr_no_hit() -> None:
    assert mrr(["hr_a", "legal_a"], {"banking"}) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    retrieved = ["banking_a", "banking_b", "hr_a", "legal_a"]
    assert abs(ndcg_at_k(retrieved, {"banking"}, 4) - 1.0) < 1e-9


def test_ndcg_never_exceeds_one_with_multiple_matches() -> None:
    # Regression test: an earlier version normalized by len(relevant_ids) (a domain-prefix
    # count) rather than actual hits found, which let nDCG exceed 1.0 whenever more than
    # one retrieved item matched the same prefix.
    retrieved = ["banking_a", "hr_a", "banking_b", "legal_a"]
    score = ndcg_at_k(retrieved, {"banking"}, 4)
    assert 0.0 <= score <= 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect() -> None:
    relevant = {"banking"}
    perfect = ndcg_at_k(["banking_a", "banking_b", "hr_a", "legal_a"], relevant, 4)
    worst = ndcg_at_k(["hr_a", "legal_a", "banking_a", "banking_b"], relevant, 4)
    assert worst < perfect
    assert worst > 0.0


def test_ndcg_no_relevant_ids_returns_zero() -> None:
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


def test_ndcg_empty_retrieved_returns_zero() -> None:
    assert ndcg_at_k([], {"banking"}, 5) == 0.0


def test_context_precision_rewards_early_relevant_hits() -> None:
    relevant = {"banking"}
    early = context_precision(["banking_a", "hr_a", "banking_b"], relevant)
    late = context_precision(["hr_a", "banking_a", "banking_b"], relevant)
    assert early > late


def test_context_precision_no_hits_is_zero() -> None:
    assert context_precision(["hr_a", "legal_a"], {"banking"}) == 0.0


def test_context_recall_matches_full_list_recall() -> None:
    retrieved = ["hr_a", "banking_a", "legal_a"]
    assert context_recall(retrieved, {"banking"}) == recall_at_k(
        retrieved, {"banking"}, len(retrieved)
    )
