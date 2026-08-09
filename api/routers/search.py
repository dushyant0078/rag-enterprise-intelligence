from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_request_id, get_state, require_api_key, resolve_domain_filters
from api.schemas import RetrievedPassageOut, SearchRequest, SearchResponse
from api.state import AppState
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> SearchResponse:
    """Retrieval only -- no LLM call, no citation validation. Useful for a
    "source explorer" UI or for callers that want raw passages, not a
    synthesized answer. `rerank=true` runs the same LexicalOverlapReranker
    the conversational pipeline uses, letting a caller directly compare
    dense-only vs. reranked ordering for the same query (see
    docs/06_reranking_and_prompting.md's diagnosed example of what that
    changes and why).
    """
    filters = resolve_domain_filters(body.domain, auth.allowed_domains)

    results = state.rag_pipeline.retriever.retrieve(body.query, top_k=body.top_k, filters=filters)
    if body.rerank and results:
        if state.rag_pipeline.reranker is None:
            raise HTTPException(
                status_code=503, detail="No reranker is configured on this deployment"
            )
        results = state.rag_pipeline.reranker.rerank(body.query, results, top_k=body.top_k)

    log_request(request_id=request_id, user_id=auth.user_id, endpoint="/search", query=body.query)

    return SearchResponse(results=[RetrievedPassageOut(**r.model_dump()) for r in results])
