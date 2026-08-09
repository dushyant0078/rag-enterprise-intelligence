from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_state, require_api_key
from api.schemas import MonitoringSummaryResponse
from api.state import AppState
from src.security.api_auth import ApiKeyRecord

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/summary", response_model=MonitoringSummaryResponse)
def monitoring_summary(
    state: AppState = Depends(get_state), _auth: ApiKeyRecord = Depends(require_api_key)
) -> MonitoringSummaryResponse:
    """Token usage, estimated cost, live retrieval health, the rolling
    hallucination rate (plus any recent alerts), and feedback satisfaction
    -- the spec's Milestone 11 monitoring items, aggregated from real
    traffic through /chat and /query (see MonitoringAggregator.record_response,
    called from those routers), not synthesized for this endpoint.
    """
    return MonitoringSummaryResponse(**state.monitoring.snapshot())
