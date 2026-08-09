from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_request_id, get_state, require_api_key, resolve_domain_filters
from api.schemas import AnswerResponse, CitationOut, QueryRequest, RetrievedPassageOut
from api.state import AppState
from src.prompting.registry import get_template
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request
from src.security.prompt_injection import detect_injection_attempt

router = APIRouter(tags=["query"])


@router.post("/query", response_model=AnswerResponse)
def query(
    body: QueryRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> AnswerResponse:
    """Stateless single-turn RAG: no session, no conversation history --
    each call is independent (see /chat for the conversational version).
    """
    if detect_injection_attempt(body.query):
        log_request(
            request_id=request_id, user_id=auth.user_id, endpoint="/query",
            query=body.query, status="rejected_injection_attempt",
        )  # fmt: skip
        raise HTTPException(
            status_code=400, detail="Request rejected: suspicious instruction pattern detected"
        )

    filters = resolve_domain_filters(body.domain, auth.allowed_domains)

    session_id = state.rag_pipeline.start_session(user_id=auth.user_id)
    result = state.rag_pipeline.ask(
        session_id,
        body.query,
        requesting_user_id=auth.user_id,
        filters=filters,
        prompt_template=get_template(body.template),
    )

    log_request(request_id=request_id, user_id=auth.user_id, endpoint="/query", query=body.query)
    state.monitoring.record_response(body.query, result)

    return AnswerResponse(
        answer=result.answer_text,
        is_grounded=result.is_grounded,
        confidence_score=result.confidence_score,
        citations=[CitationOut(**c.model_dump()) for c in result.citations],
        retrieved_passages=[
            RetrievedPassageOut(**p.model_dump()) for p in result.retrieved_passages
        ],
    )
