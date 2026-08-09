"""PII masking for audit logs (the spec's "Sensitive data masking" / "PII
detection" items). Applied only to what gets *logged* -- never to the text
handed to the retrieval/generation pipeline, which needs the real query to
function. Regex-based pattern matching, not a trained PII model: real and
useful for the common, high-confidence patterns below, not a claim of
comprehensive PII detection (names, addresses, and other unstructured PII
need a different approach entirely).
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
}


def mask_pii(text: str) -> str:
    masked = text
    for label, pattern in _PATTERNS.items():
        masked = pattern.sub(f"[REDACTED_{label}]", masked)
    return masked
