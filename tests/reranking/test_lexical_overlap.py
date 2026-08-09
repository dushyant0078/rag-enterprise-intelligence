from __future__ import annotations

from src.reranking.base import BaseReranker
from src.reranking.lexical_overlap import LexicalOverlapReranker, _lexical_score
from src.retrieval.schemas import RetrievedChunk
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.vectorstore.schemas import VectorRecord


def _chunk(chunk_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=chunk_id, text=text, score=score, domain="test")


def test_phrase_match_scores_higher_than_scattered_keywords() -> None:
    query = "enhanced due diligence wire transfers"
    off_target = (
        "Wire transfers are processed daily and reviewed by the operations team for general "
        "banking compliance, fraud monitoring, and audit purposes across account types."
    )
    on_target = "Enhanced due diligence is required for wire transfers over ten thousand dollars."
    assert _lexical_score(query, on_target) > _lexical_score(query, off_target)


def test_trigram_match_scores_higher_than_incidental_bigram_match() -> None:
    query = "enhanced due diligence wire transfer"
    # Same bag of words as the trigram-match text below, permuted so only an incidental
    # 2-word overlap ("diligence wire") survives -- not the full 3-word phrase.
    bigram_only = "diligence wire enhanced transfer due review quarterly"
    trigram_match = "enhanced due diligence wire transfer review quarterly"
    assert _lexical_score(query, trigram_match) > _lexical_score(query, bigram_only)


def test_reranker_breaks_a_genuine_bm25_tie() -> None:
    """BM25 is a bag-of-words model -- two chunks with the identical word multiset,
    merely permuted, score identically under it. This is a real, unmodified BM25 tie
    (not simulated), and reranking's phrase-awareness is what breaks it.
    """
    query = "enhanced due diligence wire transfer"
    chunk_a_text = "diligence wire enhanced transfer due review quarterly."
    chunk_b_text = "enhanced due diligence wire transfer review quarterly."
    distractor_text = "employee parental leave policy covers weeks of paid time off."

    records = [
        VectorRecord(
            id="a", vector=[], metadata={"text": chunk_a_text, "doc_id": "a", "domain": "banking"}
        ),
        VectorRecord(
            id="b", vector=[], metadata={"text": chunk_b_text, "doc_id": "b", "domain": "banking"}
        ),
        VectorRecord(
            id="d", vector=[], metadata={"text": distractor_text, "doc_id": "d", "domain": "hr"}
        ),
    ]
    bm25_results = BM25Retriever(records=records).retrieve(query, top_k=3)

    # Confirm this really is an unbroken tie before reranking -- the point being tested.
    scores_by_id = {r.chunk_id: r.score for r in bm25_results}
    assert scores_by_id["a"] == scores_by_id["b"]

    reranked = LexicalOverlapReranker().rerank(query, bm25_results)
    assert reranked[0].chunk_id == "b"  # the coherent phrase now correctly ranks first
    assert reranked[0].score > reranked[1].score


def test_rerank_empty_list_returns_empty() -> None:
    assert LexicalOverlapReranker().rerank("query", []) == []


def test_rerank_respects_top_k() -> None:
    chunks = [_chunk(str(i), f"chunk number {i} text") for i in range(5)]
    reranked = LexicalOverlapReranker().rerank("chunk text", chunks, top_k=2)
    assert len(reranked) == 2


def test_rerank_annotates_pre_rerank_metadata() -> None:
    chunks = [_chunk("a", "enhanced due diligence wire transfer", score=0.42)]
    reranked = LexicalOverlapReranker().rerank("enhanced due diligence wire transfer", chunks)
    assert reranked[0].metadata["pre_rerank_score"] == 0.42
    assert reranked[0].metadata["pre_rerank_rank"] == 0
    assert "lexical_overlap" in reranked[0].source_retriever


def test_lexical_overlap_reranker_is_a_base_reranker() -> None:
    assert isinstance(LexicalOverlapReranker(), BaseReranker)
