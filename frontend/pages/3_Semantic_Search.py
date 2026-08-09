"""Page: Semantic Search. POST /search -- retrieval only, no LLM call. The
rerank toggle directly demonstrates what Milestone 6 found (and diagnosed)
about when/why reranking changes the ordering, rather than just describing it.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
st.title("🔍 Semantic Search")

if not require_api_key():
    st.stop()

query = st.text_input("Search query")
col1, col2, col3 = st.columns(3)
domain = col1.text_input("Domain filter (optional)", value="")
top_k = col2.slider("Top K", min_value=1, max_value=20, value=5)
rerank = col3.checkbox("Apply reranking", value=False)

if st.button("Search", type="primary", disabled=not query):
    try:
        client = get_client()
        response = client.search(query, domain=domain or None, top_k=top_k, rerank=rerank)
        results = response["results"]
        if not results:
            st.info("No results.")
        for i, r in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"**#{i} — {r['doc_id']}** (score: {r['score']:.3f})")
                badges = []
                if r.get("domain"):
                    badges.append(f"domain: {r['domain']}")
                if r.get("page_numbers"):
                    badges.append(f"page(s): {', '.join(map(str, r['page_numbers']))}")
                if r.get("is_table"):
                    badges.append("📊 table")
                if badges:
                    st.caption(" · ".join(badges))
                st.write(r["text"])
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Search failed: {exc}")
