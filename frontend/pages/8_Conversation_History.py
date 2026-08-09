"""Page: Conversation History. GET /sessions/{session_id} -- replays a past
chat session's full turn history. Added as a Milestone 10 API extension;
session isolation (Milestone 7) is enforced server-side, so a scoped key
gets a 403 here trying to view another user's session, not a client-side
filter that could be bypassed.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Conversation History", page_icon="💭", layout="wide")
st.title("💭 Conversation History")

if not require_api_key():
    st.stop()

current_session = st.session_state.get("chat_session_id")
if current_session:
    st.caption(f"Current Chat Assistant session: `{current_session}`")

session_id = st.text_input("Session ID", value=current_session or "")

if st.button("Load history", type="primary", disabled=not session_id):
    try:
        client = get_client()
        history = client.get_session(session_id)

        st.caption(f"User: {history['user_id']} · {len(history['turns'])} turn(s)")
        if history.get("summary"):
            with st.expander("Rolling summary of compressed-out earlier turns"):
                st.write(history["summary"])

        for turn in history["turns"]:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])
                st.caption(turn["timestamp"])
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load session: {exc}")
