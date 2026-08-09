from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_request_id, get_state, require_api_key, resolve_domain_filters
from api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    ConversationTurnOut,
    RetrievedPassageOut,
    SessionHistoryResponse,
)
from api.state import AppState
from src.memory.session_memory import SessionAccessError
from src.prompting.registry import get_template
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request
from src.security.prompt_injection import detect_injection_attempt

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> ChatResponse:
    if detect_injection_attempt(body.query):
        log_request(
            request_id=request_id, user_id=auth.user_id, endpoint="/chat",
            query=body.query, status="rejected_injection_attempt",
        )  # fmt: skip
        raise HTTPException(
            status_code=400, detail="Request rejected: suspicious instruction pattern detected"
        )

    session_id = body.session_id or state.rag_pipeline.start_session(user_id=auth.user_id)

    filters = resolve_domain_filters(body.domain, auth.allowed_domains)

    try:
        result = state.rag_pipeline.ask(
            session_id,
            body.query,
            requesting_user_id=auth.user_id,
            filters=filters,
            prompt_template=get_template(body.template),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    state.monitoring.record_response(body.query, result)
    log_request(request_id=request_id, user_id=auth.user_id, endpoint="/chat", query=body.query)

    return ChatResponse(
        session_id=session_id,
        answer=result.answer_text,
        is_grounded=result.is_grounded,
        confidence_score=result.confidence_score,
        citations=[CitationOut(**c.model_dump()) for c in result.citations],
        retrieved_passages=[
            RetrievedPassageOut(**p.model_dump()) for p in result.retrieved_passages
        ],
    )


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(
    session_id: str,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
) -> SessionHistoryResponse:
    """A session's full turn history -- the Conversation History page's data
    source (Milestone 10). Isolation is enforced by SessionMemoryStore
    itself (Milestone 7), not just checked here: a non-admin caller
    requesting another user's session gets a 403, not another user's data
    with a client-side filter they could bypass.
    """
    try:
        session = state.conversation_manager.session_store.get_session(
            session_id, requesting_user_id=None if auth.allowed_domains is None else auth.user_id
        )
    except SessionAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'")

    return SessionHistoryResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        turns=[
            ConversationTurnOut(role=t.role, content=t.content, timestamp=t.timestamp.isoformat())
            for t in session.turns
        ],
        summary=session.summary,
    )
