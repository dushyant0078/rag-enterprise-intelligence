"""Page: Admin Monitoring. GET /health, GET /metrics (real Prometheus text
exposition format, Milestone 9), GET /model-info, GET /monitoring/summary
(Milestone 11: token usage, estimated cost, live retrieval health, rolling
hallucination rate + alerts, feedback satisfaction) -- an operational view
of the running backend, not a mock dashboard.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Admin Monitoring", page_icon="⚙️", layout="wide")
st.title("⚙️ Admin Monitoring")

if not require_api_key():
    st.stop()

client = get_client()

st.subheader("Health")
try:
    health = client.health()
    status = health["status"]
    if status == "ok":
        st.success(f"Status: **{status}**")
    else:
        st.warning(f"Status: **{status}**")
    cols = st.columns(len(health["components"]))
    for col, (name, value) in zip(cols, health["components"].items(), strict=True):
        col.metric(name.replace("_", " ").title(), value)
except APIClientError as exc:
    st.error(f"API error {exc.status_code}: {exc.detail}")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API: {exc}")

st.subheader("Configured components")
try:
    info = client.model_info()
    cols2 = st.columns(4)
    cols2[0].metric("Embedding model", info["embedding_model"])
    cols2[1].metric("LLM", info["llm_model"])
    cols2[2].metric("Vector store", info["vector_store"])
    cols2[3].metric("Reranker", info["reranker"])
    st.caption(f"Chunking strategy: {info['chunking_strategy']}")
except APIClientError as exc:
    st.error(f"API error {exc.status_code}: {exc.detail}")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API: {exc}")

st.subheader("Monitoring summary")
st.caption(
    "Live aggregates from real traffic through /chat and /query -- token usage, cost, "
    "retrieval health, hallucination rate, and feedback satisfaction (Milestone 11)."
)
try:
    summary = client.monitoring_summary()

    token_col, cost_col, retrieval_col, halluc_col, feedback_col = st.columns(5)
    token_col.metric("Total requests", summary["token_usage"]["total_requests"])
    cost_col.metric("Estimated cost", f"${summary['estimated_cost_usd']:.4f}")
    retrieval_col.metric("Empty result rate", f"{summary['retrieval']['empty_result_rate']:.0%}")
    halluc_col.metric("Grounded rate", f"{summary['hallucination']['grounded_rate']:.0%}")
    satisfaction = summary["feedback"]["satisfaction_rate"]
    feedback_col.metric(
        "Feedback satisfaction",
        f"{satisfaction:.0%}" if summary["feedback"]["total"] else "no feedback yet",
    )

    if summary["token_usage"]["by_model"]:
        st.caption("Token usage by model")
        st.json(summary["token_usage"]["by_model"])

    alerts = summary["hallucination"]["recent_alerts"]
    if alerts:
        st.warning(f"{len(alerts)} recent hallucination alert(s):")
        for alert in alerts:
            st.markdown(f"- **{alert['severity'].upper()}**: {alert['message']}")
    else:
        st.caption("No active hallucination alerts.")
except APIClientError as exc:
    st.error(f"API error {exc.status_code}: {exc.detail}")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API: {exc}")

st.subheader("Raw Prometheus metrics")
st.caption(
    "Real /metrics output (text exposition format) -- scrapeable by an actual Prometheus "
    "instance."
)
try:
    st.code(client.metrics_text(), language="text")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not fetch metrics: {exc}")
