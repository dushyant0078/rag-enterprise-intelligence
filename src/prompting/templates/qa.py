"""General question-answering over retrieved enterprise document context."""

from __future__ import annotations

from src.prompting.guardrails import INSUFFICIENT_EVIDENCE_PHRASE, STANDARD_QA_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

QA_TEMPLATE = PromptTemplate(
    name="qa",
    description="Answer a question using retrieved context, with citations and a defined "
    "fallback when the context doesn't contain the answer.",
    system_prompt=(
        "You are an enterprise document assistant. You answer employee questions using only "
        "the document excerpts provided to you as context, which may come from policy manuals, "
        "contracts, financial filings, or technical documentation."
    ),
    guardrails=STANDARD_QA_GUARDRAILS,
    user_template="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
    few_shot_examples=[
        FewShotExample(
            input=(
                "Context:\n[Source: hr-policy-2025, page 4]\nEmployees with at least one year "
                "of continuous service are eligible for 26 weeks of paid parental leave. "
                "Employees in their first year are eligible for 12 weeks.\n\n"
                "Question: How many weeks of paid parental leave does a new hire in their "
                "first year get?"
            ),
            output=(
                "A new hire in their first year of employment is eligible for 12 weeks of paid "
                "parental leave [Source: hr-policy-2025, page 4]."
            ),
        ),
        FewShotExample(
            input=(
                "Context:\n[Source: hr-policy-2025, page 4]\nEmployees with at least one year "
                "of continuous service are eligible for 26 weeks of paid parental leave.\n\n"
                "Question: What is the company's policy on remote work stipends?"
            ),
            output=INSUFFICIENT_EVIDENCE_PHRASE,
        ),
    ],
)
