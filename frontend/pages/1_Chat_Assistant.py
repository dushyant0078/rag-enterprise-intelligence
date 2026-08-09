"""Page: AI Chat Assistant. Conversational RAG via POST /chat -- session_id
is kept in st.session_state across turns, so a real multi-turn conversation
(with query rewriting, Milestone 7) actually happens across reruns, not
just within one script execution. Feedback buttons (Milestone 11) post to
POST /feedback and are reflected in the Admin Monitoring page's satisfaction
rate.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.config import SELECTABLE_TEMPLATES
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Chat Assistant", page_icon="💬", layout="wide")
st.title("💬 AI Chat Assistant")

if not require_api_key():
    st.stop()

st.session_state.setdefault("chat_session_id", None)
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("chat_feedback_given", {})

with st.sidebar:
    st.subheader("Options")
    domain = st.text_input("Domain filter (optional)", value="", key="chat_domain")
    template = st.selectbox("Prompt template", SELECTABLE_TEMPLATES, index=0, key="chat_template")
    if st.button("New conversation", use_container_width=True):
        st.session_state["chat_session_id"] = None
        st.session_state["chat_messages"] = []
        st.session_state["chat_feedback_given"] = {}
        st.rerun()
    if st.session_state["chat_session_id"]:
        st.caption(f"Session: `{st.session_state['chat_session_id']}`")


def _render_feedback_row(message_index: int, message: dict) -> None:
    given = st.session_state["chat_feedback_given"].get(message_index)
    if given:
        st.caption(f"Feedback recorded: {'👍' if given == 'up' else '👎'} Thanks!")
        return

    col_up, col_down, _ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"feedback_up_{message_index}"):
        _submit_feedback(message_index, message, "up")
    if col_down.button("👎", key=f"feedback_down_{message_index}"):
        _submit_feedback(message_index, message, "down")


def _submit_feedback(message_index: int, message: dict, rating: str) -> None:
    try:
        client = get_client()
        client.submit_feedback(
            session_id=st.session_state["chat_session_id"],
            rating=rating,
            query=message.get("query"),
            answer_excerpt=message.get("content"),
        )
        st.session_state["chat_feedback_given"][message_index] = rating
        st.rerun()
    except APIClientError as exc:
        st.error(f"Could not record feedback: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not record feedback: {exc}")


for i, message in enumerate(st.session_state["chat_messages"]):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("citations"):
            with st.expander(f"{len(message['citations'])} citation(s)"):
                for c in message["citations"]:
                    page = f", page {c['page_number']}" if c.get("page_number") else ""
                    st.markdown(f"- **{c['doc_id']}**{page} (confidence: {c['confidence']:.2f})")
        if message["role"] == "assistant":
            _render_feedback_row(i, message)

query = st.chat_input("Ask a question about the ingested documents...")
if query:
    st.session_state["chat_messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            client = get_client()
            result = client.chat(
                query,
                session_id=st.session_state["chat_session_id"],
                domain=domain or None,
                template=template,
            )
            st.session_state["chat_session_id"] = result["session_id"]
            st.write(result["answer"])

            badge = "✅ Grounded" if result["is_grounded"] else "⚠️ Not grounded"
            st.caption(f"{badge} — confidence {result['confidence_score']:.2f}")

            if result["citations"]:
                with st.expander(f"{len(result['citations'])} citation(s)"):
                    for c in result["citations"]:
                        page = f", page {c['page_number']}" if c.get("page_number") else ""
                        st.markdown(
                            f"- **{c['doc_id']}**{page} (confidence: {c['confidence']:.2f})"
                        )

            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                    "query": query,
                }
            )
            st.rerun()  # rerun so the feedback row renders under this new message immediately
        except APIClientError as exc:
            st.error(f"API error {exc.status_code}: {exc.detail}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
