"""API key authentication (the spec's "API authentication" + "Role-based
access control" items). An in-memory registry mapping key -> identity is a
starting point, not a production identity store -- a real deployment would
back this with a secrets manager or identity provider (see
docs/09_fastapi_backend.md), but the *enforcement hook* here is real: every
authenticated request carries a resolved user_id and allowed_domains that
MetadataAwareRetriever (Milestone 5) actually enforces, not a decorative check.
"""

from __future__ import annotations

import os
import secrets

from pydantic import BaseModel


class ApiKeyRecord(BaseModel):
    api_key: str
    user_id: str
    allowed_domains: list[str] | None = None  # None = unrestricted (e.g. an admin key)


class ApiKeyRegistry:
    def __init__(self) -> None:
        self._keys: dict[str, ApiKeyRecord] = {}

    def register(self, record: ApiKeyRecord) -> None:
        self._keys[record.api_key] = record

    def validate(self, api_key: str) -> ApiKeyRecord | None:
        return self._keys.get(api_key)

    @classmethod
    def with_demo_keys(cls) -> ApiKeyRegistry:
        """Seeds a small default set for local development: an unrestricted
        admin key and one domain-scoped key, both overridable via env vars
        so this never needs a hardcoded secret in source. Real credential
        provisioning (issuing/revoking keys per user) is out of scope here.
        """
        registry = cls()
        admin_key = os.environ.get("API_ADMIN_KEY") or f"admin-{secrets.token_hex(8)}"
        scoped_key = os.environ.get("API_HR_KEY") or f"hr-{secrets.token_hex(8)}"
        registry.register(ApiKeyRecord(api_key=admin_key, user_id="admin", allowed_domains=None))
        registry.register(
            ApiKeyRecord(api_key=scoped_key, user_id="hr-user", allowed_domains=["hr"])
        )
        return registry
