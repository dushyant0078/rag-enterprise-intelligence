from __future__ import annotations

from pathlib import Path

from src.embeddings.tfidf_backend import TfidfEmbedding
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord

_DOCS = [
    ("hr1", "hr", "Employees get twenty six weeks of paid parental leave after one year."),
    ("legal1", "legal", "The vendor agreement may be terminated with sixty days written notice."),
    (
        "banking1",
        "banking",
        "Wire transfers over ten thousand dollars require enhanced due diligence.",
    ),
]


def _build_records(model: TfidfEmbedding) -> list[VectorRecord]:
    texts = [t for _, _, t in _DOCS]
    vectors = model.embed(texts)
    return [
        VectorRecord(
            id=doc_id,
            vector=vectors[i].tolist(),
            metadata={"text": text, "doc_id": doc_id, "domain": domain},
        )
        for i, (doc_id, domain, text) in enumerate(_DOCS)
    ]


def test_dense_retriever_returns_most_relevant_first(tmp_path: Path) -> None:
    model = TfidfEmbedding(svd_dimensions=None).fit([t for _, _, t in _DOCS])
    records = _build_records(model)
    store = FaissVectorStore(persist_dir=tmp_path / "faiss", dimensions=model.dimensions)
    store.upsert(records, namespace="ns")

    retriever = DenseRetriever(embedding_model=model, vector_store=store, namespace="ns")
    results = retriever.retrieve("parental leave weeks", top_k=3)

    assert results[0].chunk_id == "hr1"
    assert results[0].domain == "hr"
    assert results[0].source_retriever == "dense"


def test_dense_retriever_applies_filters(tmp_path: Path) -> None:
    model = TfidfEmbedding(svd_dimensions=None).fit([t for _, _, t in _DOCS])
    records = _build_records(model)
    store = FaissVectorStore(persist_dir=tmp_path / "faiss", dimensions=model.dimensions)
    store.upsert(records, namespace="ns")

    retriever = DenseRetriever(embedding_model=model, vector_store=store, namespace="ns")
    results = retriever.retrieve("leave notice transfer", top_k=3, filters={"domain": "legal"})
    assert all(r.domain == "legal" for r in results)


def test_bm25_retriever_returns_most_relevant_first() -> None:
    model = TfidfEmbedding(svd_dimensions=None).fit([t for _, _, t in _DOCS])
    records = _build_records(model)
    retriever = BM25Retriever(records=records)

    results = retriever.retrieve("enhanced due diligence wire transfer", top_k=3)
    assert results[0].chunk_id == "banking1"
    assert results[0].source_retriever == "sparse_bm25"


def test_bm25_retriever_no_match_returns_empty() -> None:
    model = TfidfEmbedding(svd_dimensions=None).fit([t for _, _, t in _DOCS])
    records = _build_records(model)
    retriever = BM25Retriever(records=records)
    results = retriever.retrieve("xyzzyx nonexistent term qwerty", top_k=3)
    assert results == []


def test_bm25_retriever_applies_filters() -> None:
    model = TfidfEmbedding(svd_dimensions=None).fit([t for _, _, t in _DOCS])
    records = _build_records(model)
    retriever = BM25Retriever(records=records)
    results = retriever.retrieve("days notice", top_k=3, filters={"domain": "banking"})
    assert results == []  # "days"/"notice" only matches the legal doc, which is filtered out


def test_bm25_empty_corpus_returns_empty() -> None:
    retriever = BM25Retriever(records=[])
    assert retriever.retrieve("anything", top_k=5) == []
