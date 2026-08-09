"""Guide a field engineer/technician through an SOP-grounded troubleshooting
or procedure question, preserving step order and safety-critical language."""

from __future__ import annotations

from src.prompting.guardrails import STANDARD_QA_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

TECHNICAL_TROUBLESHOOTING_TEMPLATE = PromptTemplate(
    name="technical_troubleshooting",
    description="Answer a procedural/troubleshooting question from SOP or technical "
    "documentation context, preserving step order and safety warnings exactly.",
    system_prompt=(
        "You are a technical support assistant for field engineers and technicians. You answer "
        "procedural and troubleshooting questions using only the Standard Operating Procedures "
        "and technical documentation provided as context. Step order and safety warnings are "
        "safety-critical: preserve them exactly as written, never reorder, paraphrase away, or "
        "omit a safety instruction."
    ),
    guardrails=[
        *STANDARD_QA_GUARDRAILS,
        "If the context describes numbered steps, present them as a numbered list in the same "
        "order. Never merge, skip, or reorder a safety-critical step.",
    ],
    user_template="SOP / technical documentation:\n{context}\n\nQuestion: {query}\n\nAnswer:",
    few_shot_examples=[
        FewShotExample(
            input=(
                "SOP / technical documentation:\n[Source: sop-014-lockout-tagout, page 1]\n"
                "1. Notify affected personnel\n2. Shut down the unit using the normal stop "
                "procedure\n3. Isolate all energy sources at the primary disconnect\n"
                "4. Apply lock and tag to the primary disconnect\n5. Verify zero energy state "
                "before beginning work\n\n"
                "Question: What are the steps before servicing the equipment?"
            ),
            output=(
                "Before servicing the equipment, per SOP-014 [Source: sop-014-lockout-tagout, "
                "page 1]:\n1. Notify affected personnel\n2. Shut down the unit using the normal "
                "stop procedure\n3. Isolate all energy sources at the primary disconnect\n"
                "4. Apply lock and tag to the primary disconnect\n5. Verify zero energy state "
                "before beginning work"
            ),
        )
    ],
)
