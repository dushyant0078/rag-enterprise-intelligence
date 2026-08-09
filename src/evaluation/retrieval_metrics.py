"""Standard information-retrieval metrics, computed against binary relevance
ground truth (a query's `relevant_doc_ids`, from a BenchmarkQuery). No LLM
or embedding model involved -- these are exact, deterministic formulas,
unlike every "offline proxy" elsewhere in this project's evaluation code.
"""

from __future__ import annotations

import math


def _is_relevant(doc_id: str, relevant_ids: set[str]) -> bool:
    """A retrieved doc_id counts as relevant if it *starts with* any relevant
    id -- this project's doc_ids are domain-prefixed
    (e.g. "banking__aml-policy__b34a9ba2"), and benchmark ground truth is
    written at the domain-prefix level (e.g. "banking"), matching how
    Milestones 4-7's eval sets were already defined.
    """
    return any(doc_id.startswith(rel) for rel in relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(_is_relevant(doc_id, relevant_ids) for doc_id in top_k)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    # A benchmark's relevant_ids are prefixes; "recall" here means "was at least one
    # relevant item retrieved" per relevant prefix, since we don't have an exhaustive
    # list of every relevant chunk/doc -- just the domains a correct answer should draw from.
    found = {rel for rel in relevant_ids if any(_is_relevant(d, {rel}) for d in top_k)}
    return len(found) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if _is_relevant(doc_id, relevant_ids):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@K: DCG using relevance in {0, 1}, normalized by
    the ideal DCG (all relevant items packed at the top).

    Ideal-DCG's relevant-item count is the number of *actually relevant*
    items found within top_k, not len(relevant_ids) -- relevant_ids is a set
    of domain prefixes, and prefix matching means an unbounded number of
    retrieved items can match one prefix (see _is_relevant), so there's no
    fixed external "total relevant count" the way classic nDCG assumes.
    Normalizing by len(relevant_ids) instead of the actual hit count let
    nDCG exceed 1.0 whenever more than one retrieved item matched the same
    prefix -- caught by testing against a hand-computed multi-hit example,
    not assumed correct from the formula alone.
    """
    top_k = retrieved_ids[:k]
    if not top_k or not relevant_ids:
        return 0.0

    relevance_flags = [_is_relevant(doc_id, relevant_ids) for doc_id in top_k]
    dcg = sum(
        (1.0 if is_rel else 0.0) / math.log2(rank + 1)
        for rank, is_rel in enumerate(relevance_flags, start=1)
    )
    ideal_hits = sum(relevance_flags)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """RAGAS-style Context Precision: the mean of Precision@k evaluated at
    every rank k whose item is itself relevant (a form of Average Precision) --
    rewards relevant items appearing *early*, not just being present somewhere
    in the retrieved set, which plain Precision@K (fixed k) doesn't capture.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevance_flags = [_is_relevant(doc_id, relevant_ids) for doc_id in retrieved_ids]
    if not any(relevance_flags):
        return 0.0

    precisions_at_hits = []
    hits_so_far = 0
    for rank, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            hits_so_far += 1
            precisions_at_hits.append(hits_so_far / rank)
    return sum(precisions_at_hits) / len(precisions_at_hits)


def context_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """RAGAS-style Context Recall: same underlying question as recall_at_k,
    evaluated over the *entire* retrieved set rather than a fixed top-k
    cutoff -- "did retrieval find every relevant domain at all," independent
    of how deep in the ranking it took to find it.
    """
    return recall_at_k(retrieved_ids, relevant_ids, k=len(retrieved_ids))
