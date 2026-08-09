"""Page: Retrieval Explorer. Runs the same query through /search twice
(dense-only, then reranked) and shows both rankings side by side -- an
interactive version of Milestone 6's `compare_reranking.py`, over whatever
corpus is actually indexed right now instead of a fixed report.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Retrieval Explorer", page_icon="🔬", layout="wide")
st.title("🔬 Retrieval Explorer")
st.caption("Compares dense-only retrieval against the same query, reranked.")

if not require_api_key():
    st.stop()

query = st.text_input("Query")
top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

if st.button("Compare", type="primary", disabled=not query):
    try:
        client = get_client()
        with st.spinner("Retrieving..."):
            dense = client.search(query, top_k=top_k, rerank=False)["results"]
            reranked = client.search(query, top_k=top_k, rerank=True)["results"]

        col_dense, col_reranked = st.columns(2)
        with col_dense:
            st.subheader("Dense only")
            for i, r in enumerate(dense, start=1):
                st.markdown(f"**#{i}** `{r['doc_id']}` — score {r['score']:.3f}")
                st.caption(r["text"][:150] + ("..." if len(r["text"]) > 150 else ""))

        with col_reranked:
            st.subheader("Reranked")
            for i, r in enumerate(reranked, start=1):
                st.markdown(f"**#{i}** `{r['doc_id']}` — score {r['score']:.3f}")
                st.caption(r["text"][:150] + ("..." if len(r["text"]) > 150 else ""))

        dense_order = [r["chunk_id"] for r in dense]
        reranked_order = [r["chunk_id"] for r in reranked]
        if dense_order != reranked_order:
            st.info("Reranking changed the ordering for this query.")
        else:
            st.info("Reranking did not change the ordering for this query.")
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Comparison failed: {exc}")
