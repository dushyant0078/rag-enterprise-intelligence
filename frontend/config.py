"""Shared frontend configuration: page metadata and the domain list shown in
dropdowns across pages (kept in sync with the sample corpus's actual
domains -- see scripts/generate_sample_documents.py -- since Streamlit has
no clean way to query the backend for this before a page renders its
sidebar without an extra round trip on every single page).
"""

from __future__ import annotations

APP_TITLE = "RAG Enterprise Intelligence"

# Matches scripts/generate_sample_documents.py's domains. A real deployment would fetch this
# from the backend (e.g. a future GET /domains) rather than hardcode it; kept simple here since
# every page already has a free-text domain override for anything not in this list.
KNOWN_DOMAINS = [
    "banking",
    "hr",
    "legal",
    "insurance",
    "research",
    "technical",
    "financial",
    "scanned",
]

SELECTABLE_TEMPLATES = [
    "qa",
    "contract_analysis",
    "compliance_checking",
    "risk_extraction",
    "financial_analysis",
    "technical_troubleshooting",
    "research_assistance",
]
