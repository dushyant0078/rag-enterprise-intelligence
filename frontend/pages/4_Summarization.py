"""Page: Summarization. POST /summarize -- either raw pasted text or an
ingested document by id."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Summarization", page_icon="📝", layout="wide")
st.title("📝 Summarization")

if not require_api_key():
    st.stop()

source_mode = st.radio("Summarize", ["Pasted text", "Ingested document (by ID)"], horizontal=True)

text_value = None
doc_id_value = None
if source_mode == "Pasted text":
    text_value = st.text_area("Text to summarize", height=200)
else:
    doc_id_value = st.text_input(
        "Document ID", help="See the Source Viewer or Semantic Search page for real doc_id values."
    )

length_hint = st.text_input("Length hint", value="3-4 sentences")

if st.button("Summarize", type="primary", disabled=not (text_value or doc_id_value)):
    try:
        client = get_client()
        with st.spinner("Summarizing..."):
            result = client.summarize(
                text=text_value or None, doc_id=doc_id_value or None, length_hint=length_hint
            )
        st.caption(f"Source: {result['source']}")
        st.write(result["summary"])
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Summarization failed: {exc}")
