"""Landing page for the Streamlit multi-page app (`streamlit run frontend/app.py`).
Sets `api_base_url` / `api_key` in `st.session_state`, which every page under
`frontend/pages/` reads via `frontend.state.get_client()` -- Streamlit shares
session_state across pages within one browser session automatically.
"""



from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from frontend.api_client import APIClientError, resolve_default_base_url
from frontend.config import APP_TITLE
from frontend.state import get_client

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.session_state.setdefault("api_base_url", resolve_default_base_url())
st.session_state.setdefault("api_key", "")

with st.sidebar:
    st.header("API Connection")
    st.session_state["api_base_url"] = st.text_input(
        "API base URL", value=st.session_state["api_base_url"]
    )
    st.session_state["api_key"] = st.text_input(
        "API key",
        value=st.session_state["api_key"],
        type="password",
        help="See docs/09_fastapi_backend.md for demo credentials, or check the running "
        "API process's startup output / ApiKeyRegistry for a generated one.",
    )
    check_clicked = st.button("Check connection", use_container_width=True)

st.title(f"📚 {APP_TITLE}")
st.markdown(
    "Citation-aware, evaluation-driven Retrieval-Augmented Generation over enterprise "
    "document collections. Every page here is a real client of the FastAPI backend "
    "(Milestone 9) — nothing on this side talks to the pipeline directly."
)

if check_clicked or st.session_state.get("api_key"):
    try:
        client = get_client()
        health = client.health()
        status = health["status"]
        icon = "✅" if status == "ok" else "⚠️"
        if status == "ok":
            st.success(f"{icon} Connected — status: **{status}**")
        else:
            st.warning(f"{icon} Connected — status: **{status}** (no documents indexed yet)")
        cols = st.columns(len(health["components"]))
        for col, (name, value) in zip(cols, health["components"].items(), strict=True):
            col.metric(name.replace("_", " ").title(), value)
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001 -- surface connection errors plainly to the user
        st.error(f"Could not reach the API at {st.session_state['api_base_url']}: {exc}")

st.divider()
st.subheader("Pages")
st.markdown("""
- **💬 Chat Assistant** — conversational RAG with citations and session memory
- **📤 Document Upload** — ingest new documents into a domain
- **🔍 Semantic Search** — raw retrieval, with an optional reranking toggle
- **📝 Summarization** — summarize raw text or an ingested document
- **✅ Compliance Analyzer** — the real `compliance_checking` template, verdict + reasoning
- **🔬 Retrieval Explorer** — compare retrieval strategies and scores side by side
- **📄 Source Viewer** — inspect a document's full parsed content and metadata
- **💭 Conversation History** — replay a past chat session's full turn history
- **📊 Evaluation Dashboard** — run the Milestone 8 benchmark on demand
- **⚙️ Admin Monitoring** — health, Prometheus metrics, and configured components
""")
