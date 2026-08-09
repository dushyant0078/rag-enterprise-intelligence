"""Builds a ConversationalRAGPipeline from the currently-ingested corpus
(data/processed/) -- the "load corpus, chunk, embed, index, wire up a
retriever" sequence that scripts/build_index_and_compare_retrieval.py,
compare_reranking.py, run_conversational_pipeline_demo.py, and
run_evaluation_framework.py (Milestones 5-8) each wrote independently.
The API layer (Milestone 9) is a fifth consumer of the same sequence --
past the point where copy-pasting it again is reasonable, so it's
consolidated here. The milestone scripts are left as they were written
(each is a snapshot of that milestone's own demo), not retrofitted.
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

from src.chunking.registry import get_chunker
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.llm.base import BaseLLM
from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.memory.rag_pipeline import ConversationalRAGPipeline
from src.parsing.schemas import ParsedDocument
from src.prompting.registry import get_template
from src.reranking.lexical_overlap import LexicalOverlapReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.vectorstore.schemas import VectorRecord

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_INDEX_DIR = Path("data/embeddings/faiss_index")
DEFAULT_SVD_DIMENSIONS = 64


def load_processed_documents(processed_dir: Path) -> list[tuple[str, str, ParsedDocument]]:
    docs = []
    for f in sorted(glob.glob(str(processed_dir / "**" / "*.json"), recursive=True)):
        data = json.loads(Path(f).read_text())
        doc_id = Path(f).stem
        domain = doc_id.split("__")[0]
        docs.append((doc_id, domain, ParsedDocument(**data)))
    return docs


def build_index(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunking_strategy: str = "hybrid",
) -> tuple[FaissVectorStore, TfidfEmbedding, int]:
    """Returns (vector_store, embedding_model, chunk_count). Rebuilds the
    index from whatever is currently in processed_dir -- callers that need
    this to reflect newly-ingested documents should call this again (see
    the API's /reindex endpoint).

    Always a *full* rebuild, never incremental: TF-IDF's fitted vocabulary
    (and therefore its embedding dimensionality) can change whenever the
    underlying document set changes, so any previously-persisted index at
    index_dir is cleared first. Loading it and upserting fresh-dimensioned
    vectors into it raised a FAISS dimension-mismatch assertion the first
    time this was actually exercised end-to-end (via the API's /reindex
    endpoint after an upload changed the corpus) -- caught by running it,
    not by unit tests against a static corpus that never saw a dimension
    change.
    """
    docs = load_processed_documents(processed_dir)
    if not docs:
        raise FileNotFoundError(
            f"No processed documents found under {processed_dir} -- ingest documents first."
        )

    if index_dir.exists():
        shutil.rmtree(index_dir)

    chunker = get_chunker(chunking_strategy)
    chunks = []
    for doc_id, domain, doc in docs:
        for c in chunker.chunk(doc, doc_id):
            chunks.append((c, domain))

    model = TfidfEmbedding(svd_dimensions=DEFAULT_SVD_DIMENSIONS).fit([c.text for c, _ in chunks])
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

    store = FaissVectorStore(persist_dir=index_dir, dimensions=model.dimensions)
    store.upsert(records, namespace="all")
    store.persist()
    return store, model, len(records)


def build_default_pipeline(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    llm: BaseLLM | None = None,
    retrieval_top_k: int = 5,
    rerank_top_k: int = 3,
) -> ConversationalRAGPipeline:
    """`llm` defaults to DeterministicTestLLM -- the fully offline stand-in
    (see src/llm/deterministic_test_llm.py) -- so this factory works with no
    configuration. Pass a real backend (ClaudeLLM, etc.) for production use.
    """
    store, model, _ = build_index(processed_dir, index_dir)
    retriever = DenseRetriever(embedding_model=model, vector_store=store, namespace="all")
    return ConversationalRAGPipeline(
        retriever=retriever,
        llm=llm or DeterministicTestLLM(),
        prompt_template=get_template("qa"),
        reranker=LexicalOverlapReranker(),
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
