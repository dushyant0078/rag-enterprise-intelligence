"""Application state: every resource the API needs, built once at startup
(see main.py's lifespan context manager) and reused across requests --
rebuilding a TF-IDF model or FAISS index per-request would be both slow and
pointless. `AppState.reindex()` is the one operation that rebuilds it, for
the /reindex endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.citations.response_validator import ResponseValidator
from src.embeddings.tfidf_backend import TfidfEmbedding
from src.evaluation.evaluator import RAGEvaluator
from src.ingestion.pipeline import IngestionPipeline
from src.llm.base import BaseLLM
from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.memory.conversation_manager import ConversationManager
from src.memory.pipeline_factory import DEFAULT_INDEX_DIR, DEFAULT_PROCESSED_DIR, build_index
from src.memory.rag_pipeline import ConversationalRAGPipeline
from src.memory.session_memory import SessionMemoryStore
from src.monitoring.aggregator import MonitoringAggregator
from src.prompting.registry import get_template
from src.reranking.lexical_overlap import LexicalOverlapReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.vectorstore.faiss_store import FaissVectorStore
from src.security.api_auth import ApiKeyRegistry
from src.security.rate_limiter import TokenBucketRateLimiter


@dataclass
class AppState:
    processed_dir: Path
    index_dir: Path
    raw_upload_dir: Path
    metadata_dir: Path
    vector_store: FaissVectorStore
    embedding_model: TfidfEmbedding
    llm: BaseLLM
    conversation_manager: ConversationManager
    rag_pipeline: ConversationalRAGPipeline
    evaluator: RAGEvaluator
    ingestion_pipeline: IngestionPipeline
    api_key_registry: ApiKeyRegistry
    rate_limiter: TokenBucketRateLimiter
    monitoring: MonitoringAggregator
    chunk_count: int
    domains: set[str] = field(default_factory=set)

    def retriever(self) -> DenseRetriever:
        return DenseRetriever(
            embedding_model=self.embedding_model, vector_store=self.vector_store, namespace="all"
        )

    def reindex(self, domain_filter: str | None = None) -> int:
        """Rebuild the chunk index from data/processed/ -- called after
        /upload (which only ingests; indexing is a separate, explicit step,
        since re-embedding the whole corpus per upload would be wasteful at
        any real scale). `domain_filter` is accepted for API-shape
        consistency; the underlying index rebuild is always whole-corpus in
        this implementation (see docs/09_fastapi_backend.md for why a true
        per-domain incremental reindex is a documented extension, not built
        here).
        """
        store, model, count = build_index(self.processed_dir, self.index_dir)
        self.vector_store = store
        self.embedding_model = model
        self.chunk_count = count

        new_retriever = self.retriever()
        self.rag_pipeline.retriever = new_retriever
        return count


def build_app_state(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    raw_upload_dir: Path = Path("data/raw/api_uploads"),
    metadata_dir: Path = Path("data/metadata"),
    feedback_dir: Path = Path("data/feedback"),
    llm: BaseLLM | None = None,
) -> AppState:
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_upload_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    resolved_llm = llm or DeterministicTestLLM()

    try:
        store, model, chunk_count = build_index(processed_dir, index_dir)
        domains = {p.parent.name for p in processed_dir.glob("*/*.json")}
    except FileNotFoundError:
        # No documents ingested yet -- start with an empty, valid index rather than failing
        # app startup; /upload + /reindex populate it, and every read endpoint handles an
        # empty corpus gracefully (empty search results, "insufficient evidence" answers).
        model = TfidfEmbedding(svd_dimensions=64).fit(["placeholder document for an empty index"])
        store = FaissVectorStore(persist_dir=index_dir, dimensions=model.dimensions)
        chunk_count = 0
        domains = set()

    retriever = DenseRetriever(embedding_model=model, vector_store=store, namespace="all")
    conversation_manager = ConversationManager(session_store=SessionMemoryStore(), llm=resolved_llm)
    rag_pipeline = ConversationalRAGPipeline(
        retriever=retriever,
        llm=resolved_llm,
        prompt_template=get_template("qa"),
        reranker=LexicalOverlapReranker(),
        conversation_manager=conversation_manager,
        response_validator=ResponseValidator(),
        retrieval_top_k=5,
        rerank_top_k=3,
    )

    return AppState(
        processed_dir=processed_dir,
        index_dir=index_dir,
        raw_upload_dir=raw_upload_dir,
        metadata_dir=metadata_dir,
        vector_store=store,
        embedding_model=model,
        llm=resolved_llm,
        conversation_manager=conversation_manager,
        rag_pipeline=rag_pipeline,
        evaluator=RAGEvaluator(rag_pipeline, k=5),
        ingestion_pipeline=IngestionPipeline(
            processed_dir=processed_dir, metadata_dir=metadata_dir
        ),
        api_key_registry=ApiKeyRegistry.with_demo_keys(),
        rate_limiter=TokenBucketRateLimiter(capacity=60, refill_per_second=1.0),
        monitoring=MonitoringAggregator(feedback_dir=feedback_dir),
        chunk_count=chunk_count,
        domains=domains,
    )
