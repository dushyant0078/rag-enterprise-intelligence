"""Run the full evaluation framework against the real sample corpus -- the
Milestone 8 capstone, formalizing what Milestones 4-7's individual
comparison scripts each did ad hoc into one comprehensive, metric-complete
benchmark run.

Builds the real pipeline (hybrid chunking, TF-IDF embeddings, FAISS,
LexicalOverlapReranker, DeterministicTestLLM -- see
docs/08_evaluation_framework.md for why these and not the untestable
production backends), runs RAGEvaluator against the labeled 7-query
benchmark, and writes a complete report covering every metric named in the
spec: Precision@K, Recall@K, MRR, nDCG, Context Precision, Context Recall,
Citation Accuracy, Hallucination Rate, Faithfulness, Answer Correctness,
and Response Latency.

Run: python -m scripts.run_evaluation_framework
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from src.chunking.registry import get_chunker
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.evaluation.benchmark_dataset import BenchmarkDataset
from src.evaluation.evaluator import RAGEvaluator
from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.memory.rag_pipeline import ConversationalRAGPipeline
from src.parsing.schemas import ParsedDocument
from src.prompting.registry import get_template
from src.reranking.lexical_overlap import LexicalOverlapReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/embeddings/faiss_index")
REPORT_DIR = Path("reports")
K = 3


def build_pipeline() -> ConversationalRAGPipeline:
    docs = []
    for f in sorted(glob.glob(str(PROCESSED_DIR / "**" / "*.json"), recursive=True)):
        data = json.loads(Path(f).read_text())
        doc_id = Path(f).stem
        domain = doc_id.split("__")[0]
        docs.append((doc_id, domain, ParsedDocument(**data)))
    if not docs:
        raise SystemExit("No processed documents found -- run ingestion first (see Milestone 3).")

    chunker = get_chunker("hybrid")
    chunks = []
    for doc_id, domain, doc in docs:
        for c in chunker.chunk(doc, doc_id):
            chunks.append((c, domain))

    model = TfidfEmbedding(svd_dimensions=64).fit([c.text for c, _ in chunks])
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

    store = FaissVectorStore(persist_dir=INDEX_DIR, dimensions=model.dimensions)
    store.upsert(records, namespace="all")
    store.persist()

    retriever = DenseRetriever(embedding_model=model, vector_store=store, namespace="all")
    return ConversationalRAGPipeline(
        retriever=retriever,
        llm=DeterministicTestLLM(),
        prompt_template=get_template("qa"),
        reranker=LexicalOverlapReranker(),
        retrieval_top_k=5,
        rerank_top_k=K,
    )


def main() -> None:
    pipeline = build_pipeline()
    dataset = BenchmarkDataset.default_sample_corpus()
    evaluator = RAGEvaluator(pipeline, k=K)

    print(f"Running evaluation framework: {len(dataset)} benchmark queries, k={K}\n")
    report = evaluator.evaluate(dataset)

    print("=== Retrieval metrics (exact IR formulas, no LLM) ===")
    print(f"  Precision@{K}:      {report.mean_precision_at_k:.3f}")
    print(f"  Recall@{K}:         {report.mean_recall_at_k:.3f}")
    print(f"  MRR:              {report.mean_mrr:.3f}")
    print(f"  nDCG@{K}:           {report.mean_ndcg_at_k:.3f}")
    print(f"  Context Precision: {report.mean_context_precision:.3f}")
    print(f"  Context Recall:    {report.mean_context_recall:.3f}")

    print("\n=== Generation metrics (citation accuracy exact; others lexical proxies) ===")
    print(f"  Citation Accuracy:  {report.mean_citation_accuracy:.3f}")
    print(f"  Hallucination Rate: {report.mean_hallucination_rate:.3f}")
    print(f"  Faithfulness:       {report.mean_faithfulness:.3f}")
    print(f"  Answer Correctness: {report.mean_answer_correctness:.3f}")
    print(f"  Grounded Rate:      {report.grounded_rate:.3f}")

    print("\n=== Response latency ===")
    print(f"  p50:  {report.latency_p50_ms:.2f}ms")
    print(f"  p95:  {report.latency_p95_ms:.2f}ms")
    print(f"  mean: {report.latency_mean_ms:.2f}ms")

    print("\n=== Per-query detail ===")
    for r in report.per_query:
        grounded_mark = "✅" if r.generation_metrics.is_grounded else "❌"
        print(
            f"  [{grounded_mark}] "
            f"P@{K}={r.retrieval_metrics.precision_at_k:.2f} "
            f"MRR={r.retrieval_metrics.mrr:.2f} "
            f"cite_acc={r.generation_metrics.citation_accuracy:.2f} "
            f"faith={r.generation_metrics.faithfulness:.2f}  {r.query[:55]}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "milestone8_evaluation_report.json").write_text(report.model_dump_json(indent=2))
    write_markdown_report(report)
    print(f"\nWrote {REPORT_DIR / 'milestone8_evaluation_report.md'}")


def write_markdown_report(report) -> None:  # noqa: ANN001
    lines = [
        "# Evaluation Framework Report — Milestone 8",
        "",
        "Generated by `scripts/run_evaluation_framework.py`: the real `ConversationalRAGPipeline`",
        "(hybrid chunking, TF-IDF embeddings, FAISS, LexicalOverlapReranker,",
        "DeterministicTestLLM) run against the labeled 7-query benchmark",
        "(`src/evaluation/benchmark_dataset.py`), scoring every metric named in the spec.",
        "",
        "## Retrieval metrics (exact IR formulas)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision@{report.k} | {report.mean_precision_at_k:.3f} |",
        f"| Recall@{report.k} | {report.mean_recall_at_k:.3f} |",
        f"| MRR | {report.mean_mrr:.3f} |",
        f"| nDCG@{report.k} | {report.mean_ndcg_at_k:.3f} |",
        f"| Context Precision | {report.mean_context_precision:.3f} |",
        f"| Context Recall | {report.mean_context_recall:.3f} |",
        "",
        "## Generation metrics (citation accuracy exact; others lexical proxies -- see",
        "docs/08_evaluation_framework.md)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Citation Accuracy | {report.mean_citation_accuracy:.3f} |",
        f"| Hallucination Rate | {report.mean_hallucination_rate:.3f} |",
        f"| Faithfulness | {report.mean_faithfulness:.3f} |",
        f"| Answer Correctness | {report.mean_answer_correctness:.3f} |",
        f"| Grounded Rate | {report.grounded_rate:.3f} |",
        "",
        "## Response latency",
        "",
        "| Percentile | ms |",
        "|---|---|",
        f"| p50 | {report.latency_p50_ms:.2f} |",
        f"| p95 | {report.latency_p95_ms:.2f} |",
        f"| mean | {report.latency_mean_ms:.2f} |",
        "",
        "## Per-query detail",
        "",
        f"| Query | Grounded | P@{report.k} | MRR | Citation Acc. | Faithfulness |",
        "|---|---|---|---|---|---|",
    ]
    for r in report.per_query:
        lines.append(
            f"| {r.query} | {r.generation_metrics.is_grounded} | "
            f"{r.retrieval_metrics.precision_at_k:.2f} | {r.retrieval_metrics.mrr:.2f} | "
            f"{r.generation_metrics.citation_accuracy:.2f} | "
            f"{r.generation_metrics.faithfulness:.2f} |"
        )
    (REPORT_DIR / "milestone8_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
