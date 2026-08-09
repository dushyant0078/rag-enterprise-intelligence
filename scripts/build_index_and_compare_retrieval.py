"""Build a real index from the Milestone 3/4 sample corpus and compare
retrieval strategies against it — the Milestone 5 counterpart to
scripts/compare_chunking_strategies.py.

Uses the hybrid chunking strategy (Milestone 4's practical default) and the
TF-IDF embedding backend (the offline one — see
docs/04_chunking_and_embeddings.md and docs/05_vectorstore_and_retrieval.md
for why real HF/vector-cloud services aren't reachable in this sandbox) to
build a real FAISS index + BM25 index, then runs the same 7 hand-written
queries from Milestone 4 through dense, sparse, hybrid, and ensemble
retrieval and reports which strategy actually resolves each one.

Run: python -m scripts.build_index_and_compare_retrieval
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from src.chunking.registry import get_chunker
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.parsing.schemas import ParsedDocument
from src.retrieval.dense import DenseRetriever
from src.retrieval.ensemble import EnsembleRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse_bm25 import BM25Retriever
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/embeddings/faiss_index")
REPORT_DIR = Path("reports")

# Same eval set as Milestone 4's chunking comparison, for a direct before/after read on what
# reranked/fused retrieval adds on top of chunking strategy choice alone.
EVAL_QUERIES: list[tuple[str, str]] = [
    ("enhanced due diligence threshold for international wire transfers", "banking"),
    ("how many weeks of paid parental leave in the first year", "hr"),
    ("water damage coverage if the property was vacant", "insurance"),
    ("notice period to terminate the vendor agreement for convenience", "legal"),
    ("lockout tagout procedure before servicing the equipment", "technical"),
    ("which metric besides accuracy should be reported for imbalanced classification", "research"),
    ("what was the top risk factor reported last quarter", "financial"),
]


def load_documents() -> list[tuple[str, str, ParsedDocument]]:
    docs = []
    for f in sorted(glob.glob(str(PROCESSED_DIR / "**" / "*.json"), recursive=True)):
        data = json.loads(Path(f).read_text())
        doc_id = Path(f).stem
        domain = doc_id.split("__")[0]
        docs.append((doc_id, domain, ParsedDocument(**data)))
    return docs


def build_records(
    docs: list[tuple[str, str, ParsedDocument]],
) -> tuple[list[VectorRecord], TfidfEmbedding]:
    chunker = get_chunker("hybrid")
    chunks = []
    for doc_id, domain, doc in docs:
        for c in chunker.chunk(doc, doc_id):
            chunks.append((c, domain))

    model = TfidfEmbedding(svd_dimensions=128).fit([c.text for c, _ in chunks])
    vectors = model.embed([c.text for c, _ in chunks])

    records = [
        VectorRecord(
            id=c.chunk_id,
            vector=vectors[i].tolist(),
            metadata={
                "text": c.text,
                "doc_id": c.doc_id,
                "domain": domain,
                "page_numbers": c.page_numbers,
                "is_table": c.is_table,
            },
        )
        for i, (c, domain) in enumerate(chunks)
    ]
    return records, model


def evaluate(retriever, label: str) -> dict:
    hits_at_1 = 0
    per_query = []
    for query, expected_prefix in EVAL_QUERIES:
        results = retriever.retrieve(query, top_k=1)
        top1_doc = results[0].doc_id if results else ""
        hit = top1_doc.startswith(expected_prefix)
        hits_at_1 += hit
        per_query.append(
            {"query": query, "expected": expected_prefix, "top1_doc": top1_doc, "hit": hit}
        )
    return {
        "strategy": label,
        "hit_rate_at_1": round(hits_at_1 / len(EVAL_QUERIES), 2),
        "per_query": per_query,
    }


def main() -> None:
    docs = load_documents()
    if not docs:
        raise SystemExit("No processed documents found — run ingestion first (see Milestone 3).")
    print(f"Building index from {len(docs)} documents...")

    records, model = build_records(docs)
    print(f"{len(records)} chunks embedded (dimensions={model.dimensions})")

    store = FaissVectorStore(persist_dir=INDEX_DIR, dimensions=model.dimensions)
    store.upsert(records, namespace="all")
    store.persist()
    print(f"Indexed into FAISS at {INDEX_DIR} (namespace 'all', count={store.count('all')})")

    dense = DenseRetriever(embedding_model=model, vector_store=store, namespace="all")
    sparse = BM25Retriever(records=records)
    hybrid = HybridRetriever(dense, sparse, dense_weight=0.6, sparse_weight=0.4)
    ensemble = EnsembleRetriever([dense, sparse], weights=[1.0, 1.0])

    results = [
        evaluate(dense, "dense"),
        evaluate(sparse, "sparse_bm25"),
        evaluate(hybrid, "hybrid"),
        evaluate(ensemble, "ensemble_rrf"),
    ]

    print("\nRetrieval strategy comparison (Hit@1 over 7 hand-written queries):")
    for r in results:
        print(f"  {r['strategy']:<14} hit@1={r['hit_rate_at_1']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "milestone5_retrieval_comparison.json").write_text(json.dumps(results, indent=2))
    write_markdown_report(results)
    print(f"\nWrote {REPORT_DIR / 'milestone5_retrieval_comparison.md'}")


def write_markdown_report(results: list[dict]) -> None:
    lines = [
        "# Retrieval Strategy Comparison — Milestone 5",
        "",
        "Generated by `scripts/build_index_and_compare_retrieval.py` against the real ingested",
        "sample corpus, hybrid-chunked (Milestone 4) and indexed into a real local FAISS store,",
        "using the same 7 hand-written queries as the Milestone 4 chunking comparison.",
        "",
        "| Strategy | Hit@1 |",
        "|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['strategy']} | {r['hit_rate_at_1']} |")

    lines += [
        "",
        "## Per-query detail (hybrid)",
        "",
        "| Query | Expected | Top-1 doc | Hit |",
        "|---|---|---|---|",
    ]
    hybrid_result = next(r for r in results if r["strategy"] == "hybrid")
    for q in hybrid_result["per_query"]:
        lines.append(
            f"| {q['query']} | {q['expected']} | {q['top1_doc']} | {'✅' if q['hit'] else '❌'} |"
        )

    (REPORT_DIR / "milestone5_retrieval_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
