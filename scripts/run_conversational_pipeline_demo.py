"""Run the full conversational RAG pipeline end to end against the real
sample corpus — the Milestone 7 counterpart to the comparison scripts from
Milestones 4-6. Two things are demonstrated for real, not simulated:

1. Grounded answering with enforced citations: the same 7 hand-written
   queries from Milestones 4-6, run through the *entire* pipeline
   (retrieval -> reranking -> context building -> "generation" -> citation
   generation -> response validation), including one deliberately off-topic
   query to show the insufficient-evidence fallback actually firing.
2. A real multi-turn conversation, showing query rewriting resolve a
   pronoun-led follow-up using conversation history, and session memory
   correctly isolating a second user's session from the first.

Uses DeterministicTestLLM (see src/llm/deterministic_test_llm.py for why —
same sandbox network constraints as every LLM/reranker/embedding backend
since Milestone 4) and the offline TF-IDF/lexical-overlap backends from
Milestones 4-6.

Run: python -m scripts.run_conversational_pipeline_demo
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from src.chunking.registry import get_chunker
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.memory.rag_pipeline import ConversationalRAGPipeline
from src.memory.session_memory import SessionAccessError
from src.parsing.schemas import ParsedDocument
from src.prompting.registry import get_template
from src.reranking.lexical_overlap import LexicalOverlapReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/embeddings/faiss_index")
REPORT_DIR = Path("reports")

EVAL_QUERIES: list[tuple[str, str]] = [
    ("enhanced due diligence threshold for international wire transfers", "banking"),
    ("how many weeks of paid parental leave in the first year", "hr"),
    ("water damage coverage if the property was vacant", "insurance"),
    ("notice period to terminate the vendor agreement for convenience", "legal"),
    ("lockout tagout procedure before servicing the equipment", "technical"),
    ("which metric besides accuracy should be reported for imbalanced classification", "research"),
    ("what is the tallest mountain in south america", "NONE"),  # deliberately off-topic
]


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
        rerank_top_k=3,
    )


def demo_grounded_qa(pipeline: ConversationalRAGPipeline) -> list[dict]:
    print("=== Part 1: Grounded QA + citation enforcement across 7 queries ===\n")
    session_id = pipeline.start_session(user_id="demo-user")
    rows = []
    for query, expected_domain in EVAL_QUERIES:
        result = pipeline.ask(session_id, query)
        top_citation_doc = result.citations[0].doc_id if result.citations else None
        correct = (not result.is_grounded and expected_domain == "NONE") or (
            result.is_grounded and top_citation_doc and top_citation_doc.startswith(expected_domain)
        )
        status = "✅" if correct else "❌"
        print(
            f"[{status}] grounded={result.is_grounded!s:<5} "
            f"conf={result.confidence_score:.2f}  {query[:60]}"
        )
        rows.append(
            {
                "query": query,
                "expected_domain": expected_domain,
                "is_grounded": result.is_grounded,
                "confidence": round(result.confidence_score, 3),
                "answer": result.answer_text,
                "citations": [c.doc_id for c in result.citations],
                "fabricated_citations": [c.doc_id for c in result.fabricated_citations],
                "correct": correct,
            }
        )
    return rows


def demo_multiturn_and_isolation(pipeline: ConversationalRAGPipeline) -> dict:
    print("\n=== Part 2: Multi-turn conversation with query rewriting ===\n")
    session_id = pipeline.start_session(user_id="alice")

    r1 = pipeline.ask(
        session_id, "What is the enhanced due diligence threshold for wire transfers?"
    )
    print(f"Turn 1 answer: {r1.answer_text}")

    r2 = pipeline.ask(session_id, "What about it for international transfers?")
    rewritten = r2.metadata["rewritten_query"]
    print(f"Turn 2 rewritten query: {rewritten!r}")
    print(f"Turn 2 answer: {r2.answer_text}")

    isolation_held = False
    try:
        pipeline.conversation_manager.session_store.get_session(
            session_id, requesting_user_id="bob"
        )
    except SessionAccessError:
        isolation_held = True
    print(f"\nCross-user session isolation enforced: {isolation_held}")

    return {
        "turn1_answer": r1.answer_text,
        "turn2_query_rewritten_to": rewritten,
        "turn2_answer": r2.answer_text,
        "isolation_enforced": isolation_held,
    }


def main() -> None:
    pipeline = build_pipeline()
    qa_rows = demo_grounded_qa(pipeline)
    multiturn = demo_multiturn_and_isolation(pipeline)

    accuracy = sum(r["correct"] for r in qa_rows) / len(qa_rows)
    print(f"\nOverall correct (grounded-and-right, or correctly-refused): {accuracy:.2f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"qa_results": qa_rows, "accuracy": round(accuracy, 2), "multiturn_demo": multiturn}
    (REPORT_DIR / "milestone7_conversational_pipeline_demo.json").write_text(
        json.dumps(report, indent=2)
    )
    write_markdown_report(report)
    print(f"\nWrote {REPORT_DIR / 'milestone7_conversational_pipeline_demo.md'}")


def write_markdown_report(report: dict) -> None:
    lines = [
        "# Conversational RAG Pipeline Demo — Milestone 7",
        "",
        "Generated by `scripts/run_conversational_pipeline_demo.py` against the real ingested",
        "sample corpus, running the complete pipeline: retrieval -> reranking -> context",
        "building -> generation -> citation generation -> response validation.",
        "",
        f"**Overall correct (grounded-and-right, or correctly-refused): {report['accuracy']}**",
        "",
        "| Query | Expected | Grounded | Confidence | Citations | Correct |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["qa_results"]:
        lines.append(
            f"| {r['query']} | {r['expected_domain']} | {r['is_grounded']} | {r['confidence']} | "
            f"{', '.join(r['citations']) or '—'} | {'✅' if r['correct'] else '❌'} |"
        )

    mt = report["multiturn_demo"]
    lines += [
        "",
        "## Multi-turn conversation with query rewriting",
        "",
        f"- Turn 1 answer: {mt['turn1_answer']}",
        f"- Turn 2 rewritten query: `{mt['turn2_query_rewritten_to']}`",
        f"- Turn 2 answer: {mt['turn2_answer']}",
        f"- Cross-user session isolation enforced: **{mt['isolation_enforced']}**",
    ]
    (REPORT_DIR / "milestone7_conversational_pipeline_demo.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
