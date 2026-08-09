"""Extract explicitly stated risk factors from retrieved document text
(financial filings, compliance reports, project documentation)."""

from __future__ import annotations

from src.prompting.guardrails import STANDARD_ANALYSIS_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

RISK_EXTRACTION_TEMPLATE = PromptTemplate(
    name="risk_extraction",
    description="Extract explicitly stated risks from context as a structured list, without "
    "inferring risks the source text doesn't name.",
    system_prompt=(
        "You are a risk analysis assistant. You extract risk factors explicitly named in "
        "document excerpts provided as context — financial filings, compliance reports, audit "
        "findings. You list only risks the text actually states; you do not infer additional "
        "risks the source doesn't name, even if they seem likely."
    ),
    guardrails=[
        *STANDARD_ANALYSIS_GUARDRAILS,
        "Format each extracted risk as a bullet point: the risk, followed by its citation.",
    ],
    user_template=(
        "Context:\n{context}\n\n"
        "Extract every risk factor explicitly stated in the context relevant to: {query}\n\n"
        "Extracted risks:"
    ),
    few_shot_examples=[
        FewShotExample(
            input=(
                "Context:\n[Source: 10k-2025-q3, page 12]\nOur top risk factors this quarter "
                "include foreign currency exposure due to expanding European operations and "
                "supply chain concentration among a small number of key component suppliers.\n\n"
                "Extract every risk factor explicitly stated in the context relevant to: "
                "quarterly risk factors"
            ),
            output=(
                "- Foreign currency exposure from expanding European operations "
                "[Source: 10k-2025-q3, page 12]\n"
                "- Supply chain concentration among a small number of key component suppliers "
                "[Source: 10k-2025-q3, page 12]"
            ),
        )
    ],
)
