"""Page: Source Viewer. GET /documents/{doc_id} -- full parsed content and
metadata for one ingested document. Added as a Milestone 10 API extension
specifically so this page has real data to show, RBAC-checked like every
other domain-scoped read.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Source Viewer", page_icon="📄", layout="wide")
st.title("📄 Source Viewer")

if not require_api_key():
    st.stop()

doc_id = st.text_input(
    "Document ID", help="Find real doc_id values via Semantic Search or Retrieval Explorer."
)

if st.button("Load document", type="primary", disabled=not doc_id):
    try:
        client = get_client()
        doc = client.get_document(doc_id)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Domain", doc["domain"])
        col2.metric("Format", doc["format"])
        col3.metric("Pages", doc["page_count"])
        col4.metric("Version", doc["version"])

        badges = []
        if doc.get("language"):
            badges.append(f"language: {doc['language']}")
        if doc.get("used_ocr"):
            badges.append("📷 OCR-extracted")
        if badges:
            st.caption(" · ".join(badges))

        if doc["tables"]:
            st.subheader(f"Tables ({len(doc['tables'])})")
            for i, table in enumerate(doc["tables"], start=1):
                st.caption(
                    f"Table {i}" + (f" — {table['caption']}" if table.get("caption") else "")
                )
                st.table(table["rows"])

        st.subheader("Full text")
        st.text_area(
            "", value=doc["full_text"], height=400, disabled=True, label_visibility="collapsed"
        )
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load document: {exc}")
