"""Summarize a retrieved document or section — full-document context, not a
top-K chunk answer, since summarization needs the whole thing, not just the
most similar fragment."""

from __future__ import annotations

from src.prompting.guardrails import (
    CITATION_REQUIRED,
    GROUNDING,
    INJECTION_RESISTANCE,
    NO_SPECULATION,
)
from src.prompting.schemas import FewShotExample, PromptTemplate

SUMMARIZATION_TEMPLATE = PromptTemplate(
    name="summarization",
    description="Summarize retrieved document content faithfully, without adding claims not "
    "present in the source.",
    system_prompt=(
        "You are an enterprise document assistant. You write clear, faithful summaries of "
        "document content provided to you as context. Your summary must be substantially "
        "shorter than the source while preserving every material fact."
    ),
    guardrails=[GROUNDING, NO_SPECULATION, CITATION_REQUIRED, INJECTION_RESISTANCE],
    user_template=(
        "Context:\n{context}\n\n"
        "Summarize the above in {length_hint}. Preserve all numbers, dates, and defined terms "
        "exactly as written.\n\nSummary:"
    ),
    few_shot_examples=[
        FewShotExample(
            input=(
                "Context:\n[Source: aml-policy-2025, page 1]\nEnhanced due diligence applies to "
                "domestic wire transfers over $10,000 and international wire transfers over "
                "$3,000. Flagged accounts are reviewed quarterly by the Compliance Department. "
                "Accounts within 10% of a reporting threshold are escalated to the AML "
                "Officer.\n\n"
                "Summarize the above in 2-3 sentences."
            ),
            output=(
                "Enhanced due diligence (EDD) is triggered by domestic wire transfers over "
                "$10,000 and international wire transfers over $3,000 [Source: aml-policy-2025, "
                "page 1]. Flagged accounts undergo quarterly Compliance Department review, with "
                "accounts near a threshold (within 10%) escalated to the AML Officer "
                "[Source: aml-policy-2025, page 1]."
            ),
        )
    ],
)
