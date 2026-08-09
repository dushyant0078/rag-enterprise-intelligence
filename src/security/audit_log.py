"""Audit logging: one structured JSON line per API request, PII-masked,
using the same logging infrastructure as the rest of the pipeline
(src/utils/logging_config.py) rather than a bespoke logger. Every query,
retrieved chunk set, and generated answer being logged with a correlation
id is the audit requirement named in docs/01_business_understanding.md's
security constraints and docs/02_architecture.md's Auth Layer description.
"""

from __future__ import annotations

from typing import Any

from src.security.pii_masking import mask_pii
from src.utils.logging_config import get_logger

_logger = get_logger("audit")


def log_request(
    *,
    request_id: str,
    user_id: str,
    endpoint: str,
    query: str | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> None:
    _logger.info(
        "api_request",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "query": mask_pii(query) if query else None,
            "status": status,
            **(extra or {}),
        },
    )
