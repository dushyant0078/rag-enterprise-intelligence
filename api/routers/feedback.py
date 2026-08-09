from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_request_id, get_state, require_api_key
from api.schemas import FeedbackRequest, FeedbackResponse
from api.state import AppState
from src.security.api_auth import ApiKeyRecord
from src.security.audit_log import log_request

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    state: AppState = Depends(get_state),
    auth: ApiKeyRecord = Depends(require_api_key),
    request_id: str = Depends(get_request_id),
) -> FeedbackResponse:
    """Thumbs up/down on a chat response (the spec's "User feedback
    analytics" item, Milestone 11) -- persisted via FeedbackStore and
    reflected in GET /monitoring/summary's `feedback` section.
    """
    record = state.monitoring.feedback_store.record(
        session_id=body.session_id,
        user_id=auth.user_id,
        rating=body.rating,
        comment=body.comment,
        query=body.query,
        answer_excerpt=body.answer_excerpt,
    )

    log_request(
        request_id=request_id, user_id=auth.user_id, endpoint="/feedback",
        extra={"rating": body.rating, "session_id": body.session_id},
    )  # fmt: skip

    return FeedbackResponse(feedback_id=record.feedback_id)
