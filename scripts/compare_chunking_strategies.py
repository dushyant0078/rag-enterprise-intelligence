"""Compare every chunking strategy against the ingested sample corpus.

Two kinds of comparison, both real (nothing here is fabricated or assumed):

1. Structural stats — chunk count, size distribution, table handling —
   computed directly from each strategy's actual output.
2. A small retrieval-quality check — chunk the corpus with each strategy,
   embed with TF-IDF (the offline backend; see
   src/embeddings/tfidf_backend.py for why), run a handful of hand-written
   queries against the sample documents, and measure whether the correct
   document's chunk comes back as the top result via brute-force cosine
   similarity.

This is deliberately scoped smaller than Milestone 8's evaluation framework
(no Precision@K/nDCG/RAGAS, no vector database, no reranking) — it's a
chunking-strategy comparison, not the full RAG evaluation suite. See
docs/04_chunking_and_embeddings.md for why that's the right scope here and
what Milestone 8 adds on top.

Run: python scripts/compare_chunking_strategies.py
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import numpy as np

from src.chunking.registry import STRATEGIES_WITHOUT_DEPENDENCIES, get_chunker
from src.chunking.schemas import Chunk
from src.chunking.semantic import SemanticChunker
from src.chunking.text_utils import split_sentences
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.parsing.schemas import ParsedDocument

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

# Hand-written (query, expected doc_id prefix) pairs, one per sample domain from
# scripts/generate_sample_documents.py. Every answer is genuinely present in that
# domain's sample document — these aren't trick questions, just real lookup queries.
EVAL_QUERIES: list[tuple[str, str]] = [
    ("enhanced due diligence threshold for international wire transfers", "banking"),
    ("how many weeks of paid parental leave in the first year", "hr"),
    ("water damage coverage if the property was vacant", "insurance"),
    ("notice period to terminate the vendor agreement for convenience", "legal"),
    ("lockout tagout procedure before servicing the equipment", "technical"),
    ("which metric besides accuracy should be reported for imbalanced classification", "research"),
    ("what was the top risk factor reported last quarter", "financial"),
]


def load_documents() -> list[tuple[str, ParsedDocument]]:
    docs = []
    for f in sorted(glob.glob(str(PROCESSED_DIR / "**" / "*.json"), recursive=True)):
        data = json.loads(Path(f).read_text())
        docs.append((Path(f).stem, ParsedDocument(**data)))
    return docs


def build_semantic_chunker(docs: list[tuple[str, ParsedDocument]]) -> SemanticChunker:
    """TF-IDF's cosine-similarity distribution between sentences sits much
    lower than a dense transformer's (see docs/04_chunking_and_embeddings.md)
    — 0.1 was chosen empirically against this corpus, not copied from a
    dense-embedding default that wouldn't mean the same thing here.
    """
    all_sentences = [s for _, doc in docs for s in split_sentences(doc.full_text)]
    sentence_model = TfidfEmbedding(svd_dimensions=None).fit(all_sentences)
    return SemanticChunker(
        embed_fn=sentence_model.embed, similarity_threshold=0.1, max_chunk_size=1500
    )


def chunk_corpus(strategy: str, docs: list[tuple[str, ParsedDocument]]) -> list[Chunk]:
    chunker = build_semantic_chunker(docs) if strategy == "semantic" else get_chunker(strategy)

    chunks: list[Chunk] = []
    for doc_id, doc in docs:
        chunks.extend(chunker.chunk(doc, doc_id))
    return chunks


def structural_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"chunk_count": 0}
    sizes = [c.char_count for c in chunks]
    return {
        "chunk_count": len(chunks),
        "avg_chars": round(statistics.mean(sizes), 1),
        "median_chars": statistics.median(sizes),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "stdev_chars": round(statistics.pstdev(sizes), 1) if len(sizes) > 1 else 0.0,
        "pct_table_chunks": round(100 * sum(c.is_table for c in chunks) / len(chunks), 1),
        "pct_with_heading_metadata": round(
            100 * sum(bool(c.metadata.get("heading_path")) for c in chunks) / len(chunks), 1
        ),
    }


def retrieval_eval(chunks: list[Chunk]) -> dict:
    """Brute-force cosine-similarity top-K check against EVAL_QUERIES. A
    genuine (if small-scale) measurement — no vector database or reranking
    involved yet (that's Milestone 5), just numpy.
    """
    texts = [c.text for c in chunks]
    model = TfidfEmbedding(svd_dimensions=None).fit(texts)
    chunk_vectors = model.embed(texts)

    hits_at_1 = 0
    hits_at_3 = 0
    per_query = []
    for query, expected_prefix in EVAL_QUERIES:
        query_vector = model.embed([query])[0]
        similarities = chunk_vectors @ query_vector  # already L2-normalized -> cosine similarity
        ranked_idx = np.argsort(-similarities)

        top1_doc = chunks[ranked_idx[0]].doc_id
        top3_docs = [chunks[i].doc_id for i in ranked_idx[:3]]

        hit1 = top1_doc.startswith(expected_prefix)
        hit3 = any(d.startswith(expected_prefix) for d in top3_docs)
        hits_at_1 += hit1
        hits_at_3 += hit3
        per_query.append(
            {
                "query": query,
                "expected": expected_prefix,
                "top1_doc": top1_doc,
                "hit@1": hit1,
                "hit@3": hit3,
            }
        )

    n = len(EVAL_QUERIES)
    return {
        "hit_rate_at_1": round(hits_at_1 / n, 2),
        "hit_rate_at_3": round(hits_at_3 / n, 2),
        "per_query": per_query,
    }


def main() -> None:
    docs = load_documents()
    if not docs:
        raise SystemExit(
            "No processed documents found under data/processed/. Run "
            "`make generate-samples` then `make ingest DOMAIN=<d> INPUT=documents/<d>` first."
        )
    print(f"Comparing chunking strategies over {len(docs)} ingested documents...\n")

    strategies = (*STRATEGIES_WITHOUT_DEPENDENCIES, "semantic")
    results: dict[str, dict] = {}
    for strategy in strategies:
        chunks = chunk_corpus(strategy, docs)
        stats = structural_stats(chunks)
        eval_result = retrieval_eval(chunks)
        results[strategy] = {**stats, **eval_result}
        print(
            f"{strategy:<16} chunks={stats['chunk_count']:<4} "
            f"avg_chars={stats.get('avg_chars', 0):<7} "
            f"table%={stats.get('pct_table_chunks', 0):<6} "
            f"hit@1={eval_result['hit_rate_at_1']:<5} hit@3={eval_result['hit_rate_at_3']}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "milestone4_chunking_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_markdown_report(results)
    print(f"\nWrote {REPORT_DIR / 'milestone4_chunking_comparison.md'}")


def write_markdown_report(results: dict[str, dict]) -> None:
    lines = [
        "# Chunking Strategy Comparison — Milestone 4",
        "",
        "Generated by `scripts/compare_chunking_strategies.py` against the real ingested sample",
        "corpus from Milestone 3, using the TF-IDF offline embedding backend (see",
        "docs/04_chunking_and_embeddings.md for why, and for the scope/limitations of this",
        "comparison relative to the full evaluation framework in Milestone 8).",
        "",
        "## Structural comparison",
        "",
        "| Strategy | Chunks | Avg chars | Stdev | % table | % w/ heading |",
        "|---|---|---|---|---|---|",
    ]
    for strategy, r in results.items():
        lines.append(
            f"| {strategy} | {r['chunk_count']} | {r.get('avg_chars', 0)} | "
            f"{r.get('stdev_chars', 0)} | {r.get('pct_table_chunks', 0)} | "
            f"{r.get('pct_with_heading_metadata', 0)} |"
        )

    lines += [
        "",
        "## Retrieval-quality check (7 hand-written queries, one per sample domain)",
        "",
        "| Strategy | Hit@1 | Hit@3 |",
        "|---|---|---|",
    ]
    for strategy, r in results.items():
        lines.append(f"| {strategy} | {r['hit_rate_at_1']} | {r['hit_rate_at_3']} |")

    lines += [
        "",
        "## Per-query detail (best-performing strategy shown; see .json for all strategies)",
        "",
    ]
    best_strategy = max(
        results, key=lambda s: (results[s]["hit_rate_at_1"], results[s]["hit_rate_at_3"])
    )
    lines.append(f"Best on this small eval set: **{best_strategy}**")
    lines.append("")
    lines.append("| Query | Expected domain | Top-1 doc | Hit@1 |")
    lines.append("|---|---|---|---|")
    for q in results[best_strategy]["per_query"]:
        lines.append(
            f"| {q['query']} | {q['expected']} | {q['top1_doc']} | {'✅' if q['hit@1'] else '❌'} |"
        )

    (REPORT_DIR / "milestone4_chunking_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
