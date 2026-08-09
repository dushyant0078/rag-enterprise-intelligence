"""Shared helpers every page uses: build an API client from whatever base
URL/API key are currently in session_state (set on the landing page's
sidebar, shared automatically across pages in a real multi-page Streamlit
session), with sane defaults so a page still works if run standalone
(e.g. under AppTest, or as the very first page a user lands on directly).
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import RAGAPIClient, resolve_default_base_url


def get_client() -> RAGAPIClient:
    base_url = st.session_state.get("api_base_url", resolve_default_base_url())
    api_key = st.session_state.get("api_key", "")
    return RAGAPIClient(base_url=base_url, api_key=api_key)


def require_api_key() -> bool:
    """Shows a standard warning and returns False if no API key is set yet,
    so every page can `if not require_api_key(): st.stop()` instead of
    repeating the same check and message.
    """
    if not st.session_state.get("api_key"):
        st.warning(
            "Set an API key in the sidebar on the home page before using this page. "
            "See docs/09_fastapi_backend.md for demo credentials.",
            icon="🔑",
        )
        return False
    return True
