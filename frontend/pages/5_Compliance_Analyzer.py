"""Page: Compliance Analyzer. Uses POST /query with
`template="compliance_checking"` -- the real Milestone 6 template (verdict +
reasoning, grounded against retrieved policy text), not a relabeled QA call.
Adding template selection to /query was a Milestone 10 API extension made
specifically so this page could be real.
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIClientError
from frontend.state import get_client, require_api_key

st.set_page_config(page_title="Compliance Analyzer", page_icon="✅", layout="wide")
st.title("✅ Compliance Analyzer")
st.caption(
    "Describe a situation; the assistant checks it against retrieved policy text and states "
    "a Compliant / Non-Compliant / Insufficient Information verdict with reasoning."
)

if not require_api_key():
    st.stop()

situation = st.text_area(
    "Situation to check",
    placeholder="e.g. A $12,000 domestic wire transfer was processed without an enhanced due "
    "diligence review.",
    height=120,
)
domain = st.text_input("Domain filter (optional)", value="")

if st.button("Check compliance", type="primary", disabled=not situation):
    try:
        client = get_client()
        with st.spinner("Checking against policy..."):
            result = client.query(situation, domain=domain or None, template="compliance_checking")
        st.markdown(result["answer"])

        badge = "✅ Grounded" if result["is_grounded"] else "⚠️ Not grounded"
        st.caption(f"{badge} — confidence {result['confidence_score']:.2f}")

        if result["citations"]:
            with st.expander(f"{len(result['citations'])} citation(s)"):
                for c in result["citations"]:
                    page = f", page {c['page_number']}" if c.get("page_number") else ""
                    st.markdown(f"- **{c['doc_id']}**{page} (confidence: {c['confidence']:.2f})")
                    if c.get("excerpt"):
                        st.caption(c["excerpt"])
    except APIClientError as exc:
        st.error(f"API error {exc.status_code}: {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
