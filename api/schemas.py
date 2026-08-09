"""Request/response schemas for every endpoint. Kept separate from
src/*/schemas.py -- these are the HTTP contract, which is allowed to
reshape internal models for API stability (e.g. hiding internal fields)
even as the internals evolve underneath.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Every prompt template from Milestone 6 except "summarization" (which has a different
# render() signature -- length_hint, not a query -- and already has its own dedicated
# /summarize endpoint). Selectable on /query and /chat so pages like the Compliance
# Analyzer (Milestone 10) get the real, task-specific template, not a relabeled QA call.
SelectableTemplate = Literal[
    "qa",
    "contract_analysis",
    "compliance_checking",
    "risk_extraction",
    "financial_analysis",
    "technical_troubleshooting",
    "research_assistance",
]


class CitationOut(BaseModel):
    doc_id: str
    page_number: int | None = None
    chunk_id: str | None = None
    confidence: float
    excerpt: str


class RetrievedPassageOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    domain: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    is_table: bool = False


class AnswerResponse(BaseModel):
    answer: str
    is_grounded: bool
    confidence_score: float
    citations: list[CitationOut]
    retrieved_passages: list[RetrievedPassageOut]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    domain: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    template: SelectableTemplate = "qa"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    domain: str | None = None
    template: SelectableTemplate = "qa"


class ChatResponse(AnswerResponse):
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    domain: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = False


class SearchResponse(BaseModel):
    results: list[RetrievedPassageOut]


class SummarizeRequest(BaseModel):
    text: str | None = None
    doc_id: str | None = None
    length_hint: str = "3-4 sentences"


class SummarizeResponse(BaseModel):
    summary: str
    source: str  # "text" | "doc_id:<id>"


class IngestionResultOut(BaseModel):
    doc_id: str
    status: str
    domain: str
    message: str | None = None


class ReindexRequest(BaseModel):
    domain: str | None = None  # None = reindex every ingested domain


class ReindexResponse(BaseModel):
    status: str
    chunks_indexed: int
    domains: list[str]


class DocumentDetailResponse(BaseModel):
    doc_id: str
    domain: str
    format: str
    page_count: int
    language: str | None = None
    used_ocr: bool
    version: int
    full_text: str
    tables: list[dict[str, Any]] = Field(default_factory=list)


class ConversationTurnOut(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    user_id: str
    turns: list[ConversationTurnOut]
    summary: str | None = None


class FeedbackRequest(BaseModel):
    session_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
    query: str | None = None
    answer_excerpt: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    recorded: bool = True


class MonitoringSummaryResponse(BaseModel):
    token_usage: dict[str, Any]
    estimated_cost_usd: float
    retrieval: dict[str, Any]
    hallucination: dict[str, Any]
    feedback: dict[str, Any]


class EvaluateRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20)


class EvaluateResponse(BaseModel):
    benchmark_size: int
    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_citation_accuracy: float
    mean_faithfulness: float
    mean_answer_correctness: float
    grounded_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]


class ModelInfoResponse(BaseModel):
    embedding_model: str
    llm_model: str
    vector_store: str
    reranker: str
    chunking_strategy: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class RootResponse(BaseModel):
    name: str
    version: str
    docs_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
