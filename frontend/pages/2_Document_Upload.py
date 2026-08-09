"""Page: Document Upload. POST /upload (parse + hash + version-track) is
deliberately separate from POST /reindex (rebuild the searchable index) --
matching the API's own two-step design (Milestone 9): batch several uploads,
then reindex once, rather than re-embedding the whole corpus per file.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.config import KNOWN_DOMAINS
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Document Upload", page_icon="📤", layout="wide")
st.title("📤 Document Upload")

if not require_api_key():
    st.stop()

st.markdown("Supported formats: PDF (native + scanned/OCR), DOCX, PPTX, HTML, Markdown, CSV, TXT.")

uploaded_file = st.file_uploader("Choose a file", type=None)
domain = st.selectbox("Domain", KNOWN_DOMAINS, index=0)
custom_domain = st.text_input("...or a custom domain (overrides the dropdown)", value="")

if st.button("Upload", disabled=uploaded_file is None, type="primary"):
    if uploaded_file is None:
        # The button is disabled in this case, but Streamlit's `disabled` is a UI hint, not a
        # type guard -- checked explicitly too, both for real safety and so mypy can narrow.
        st.error("Choose a file first.")
        st.stop()

    target_domain = custom_domain.strip() or domain
    try:
        client = get_client()
        result = client.upload(uploaded_file.name, uploaded_file.getvalue(), target_domain)
        status = result["status"]
        if status == "ingested":
            st.success(f"Ingested as `{result['doc_id']}` in domain **{target_domain}**.")
        elif status == "skipped_unchanged":
            st.info(f"Unchanged since last upload — `{result['doc_id']}` already ingested.")
        elif status == "new_version":
            st.success(f"New version ingested for `{result['doc_id']}`.")
        elif status == "duplicate":
            st.warning("Duplicate content — matches an already-ingested document.")
        else:
            st.warning(f"Status: {status} — {result.get('message', '')}")
        st.caption("Documents aren't searchable until you reindex (below).")
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Upload failed: {exc}")

st.divider()
st.subheader("Reindex")
st.caption("Rebuilds the chunk/embedding/vector index from every ingested document.")
if st.button("Reindex now"):
    try:
        client = get_client()
        with st.spinner("Reindexing..."):
            result = client.reindex()
        st.success(
            f"Indexed {result['chunks_indexed']} chunks across domains: "
            f"{', '.join(result['domains']) or '(none)'}"
        )
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Reindex failed: {exc}")
