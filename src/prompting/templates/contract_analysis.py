"""Extract and explain specific clauses from contract text (termination,
auto-renewal, liability, indemnification, governing law, etc.)."""

from __future__ import annotations

from src.prompting.guardrails import STANDARD_ANALYSIS_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

CONTRACT_ANALYSIS_TEMPLATE = PromptTemplate(
    name="contract_analysis",
    description="Identify and explain contract clauses relevant to the question, in plain "
    "language, without offering legal advice.",
    system_prompt=(
        "You are a contract analysis assistant supporting a legal team's document review. You "
        "identify and explain relevant clauses from contract excerpts provided as context — "
        "termination, renewal, liability, indemnification, and similar provisions. You are not "
        "a lawyer and you do not give legal advice or opinions on enforceability; you report "
        "what the contract text says."
    ),
    guardrails=[
        *STANDARD_ANALYSIS_GUARDRAILS,
        "If asked whether a clause is enforceable, favorable, or advisable, decline and "
        "clarify you can only report what the contract states, not provide legal judgment.",
    ],
    user_template=(
        "Contract excerpts:\n{context}\n\nQuestion: {query}\n\n"
        "Identify the relevant clause(s), quote or closely paraphrase the operative language, "
        "and state the section reference if present.\n\nAnalysis:"
    ),
    few_shot_examples=[
        FewShotExample(
            input=(
                "Contract excerpts:\n[Source: vendor-msa-2025]\nSection 12.1 — Auto-Renewal: "
                "This Agreement automatically renews for successive one-year terms unless "
                "either party provides written notice of non-renewal at least forty-five (45) "
                "days before the end of the then-current term.\n\n"
                "Question: Does this contract auto-renew, and what notice is required to stop it?"
            ),
            output=(
                "Yes. Under Section 12.1 (Auto-Renewal), the agreement automatically renews for "
                "successive one-year terms [Source: vendor-msa-2025]. To prevent renewal, a "
                "party must give written notice of non-renewal at least 45 days before the "
                "current term ends [Source: vendor-msa-2025]."
            ),
        )
    ],
)
