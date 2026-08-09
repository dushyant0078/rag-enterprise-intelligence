from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.requests import Request

from api.dependencies import get_state, require_api_key
from api.schemas import HealthResponse, ModelInfoResponse, RootResponse
from api.state import AppState

router = APIRouter(tags=["system"])


@router.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        name="RAG Enterprise Intelligence API",
        version="0.1.0",
        docs_url="/docs",
        metadata={"architecture": "see docs/02_architecture.md"},
    )


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    components = {
        "vector_store": "ok" if state.chunk_count > 0 else "empty (no documents ingested yet)",
        "llm": state.llm.model_name,
        "ingestion_pipeline": "ok",
    }
    overall = "ok" if state.chunk_count > 0 else "degraded"
    return HealthResponse(status=overall, components=components)


@router.get("/metrics")
def metrics(request: Request) -> Response:
    text = request.app.state.metrics_registry.render_prometheus()
    return Response(content=text, media_type="text/plain; version=0.0.4")


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    state: AppState = Depends(get_state), _auth=Depends(require_api_key)
) -> ModelInfoResponse:
    return ModelInfoResponse(
        embedding_model=state.embedding_model.model_name,
        llm_model=state.llm.model_name,
        vector_store="faiss",
        reranker="lexical_overlap",
        chunking_strategy="hybrid",
    )
