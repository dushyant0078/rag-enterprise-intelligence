"""Page: Evaluation Dashboard. POST /evaluate -- runs the Milestone 8
evaluation framework's full labeled benchmark through the live pipeline on
demand and renders every metric group."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption(
    "Runs the full 7-query labeled benchmark through the live pipeline -- real work, not a "
    "canned report. Expect this to take a few seconds."
)

if not require_api_key():
    st.stop()

k = st.slider("K (for Precision@K / Recall@K / nDCG@K)", min_value=1, max_value=10, value=5)

if st.button("Run evaluation", type="primary"):
    try:
        client = get_client()
        with st.spinner("Running benchmark..."):
            report = client.evaluate(k=k)

        st.subheader("Retrieval metrics")
        cols = st.columns(6)
        cols[0].metric(f"Precision@{report['k']}", f"{report['mean_precision_at_k']:.3f}")
        cols[1].metric(f"Recall@{report['k']}", f"{report['mean_recall_at_k']:.3f}")
        cols[2].metric("MRR", f"{report['mean_mrr']:.3f}")
        cols[3].metric(f"nDCG@{report['k']}", f"{report['mean_ndcg_at_k']:.3f}")
        cols[4].metric("Citation Acc.", f"{report['mean_citation_accuracy']:.3f}")
        cols[5].metric("Grounded Rate", f"{report['grounded_rate']:.3f}")

        st.subheader("Generation metrics")
        st.caption(
            "Faithfulness/Answer Correctness are lexical proxies -- see "
            "docs/08_evaluation_framework.md"
        )
        cols2 = st.columns(2)
        cols2[0].metric("Faithfulness", f"{report['mean_faithfulness']:.3f}")
        cols2[1].metric("Answer Correctness", f"{report['mean_answer_correctness']:.3f}")

        st.subheader("Response latency")
        cols3 = st.columns(2)
        cols3[0].metric("p50", f"{report['latency_p50_ms']:.1f} ms")
        cols3[1].metric("p95", f"{report['latency_p95_ms']:.1f} ms")

        st.caption(f"Benchmark size: {report['benchmark_size']} queries")
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Evaluation failed: {exc}")
