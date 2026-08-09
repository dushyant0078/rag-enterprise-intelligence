"""FastAPI dependencies. Auth and rate limiting are real `Depends()` checks
that reject a request (401/429) before the route body runs, not decorative
comments -- see tests/api for requests that actually get rejected.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Header, HTTPException, Request

from api.state import AppState
from src.security.api_auth import ApiKeyRecord


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_request_id(x_request_id: str | None = Header(default=None)) -> str:
    return x_request_id or str(uuid.uuid4())


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> ApiKeyRecord:
    state: AppState = request.app.state.app_state
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    record = state.api_key_registry.validate(x_api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not state.rate_limiter.allow(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return record


def resolve_domain_filters(
    requested_domain: str | None, allowed_domains: list[str] | None
) -> dict[str, Any] | None:
    """RBAC: intersect the caller's requested domain with what their API key
    is actually allowed to see, never widen it -- same enforcement
    MetadataAwareRetriever provides directly when wrapped around a retriever
    (Milestone 5); applied here at the API boundary since this pipeline's
    retriever isn't itself RBAC-wrapped by default. Shared by every route
    that accepts a `domain` filter (chat/query/search), rather than each
    reimplementing the same intersection logic.
    """
    if allowed_domains is None:
        return {"domain": requested_domain} if requested_domain else None

    requested = [requested_domain] if requested_domain else list(allowed_domains)
    return {"domain": [d for d in requested if d in allowed_domains]}
