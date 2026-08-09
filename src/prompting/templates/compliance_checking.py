"""Check whether a described situation/action complies with retrieved
policy text — the compliance-analyzer use case from docs/02_architecture.md."""

from __future__ import annotations

from src.prompting.guardrails import STANDARD_ANALYSIS_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

COMPLIANCE_CHECKING_TEMPLATE = PromptTemplate(
    name="compliance_checking",
    description="Assess whether a described situation complies with retrieved policy text, "
    "with an explicit compliant/non-compliant/insufficient-information verdict.",
    system_prompt=(
        "You are a compliance assistant. Given a described situation and retrieved policy "
        "excerpts, you assess compliance strictly against what the policy text says. You always "
        "state a clear verdict — Compliant, Non-Compliant, or Insufficient Information — "
        "followed by your reasoning grounded in the cited policy text. You do not make final "
        "compliance determinations on the organization's behalf; you support a human reviewer's "
        "judgment, and you say so when a situation is genuinely ambiguous."
    ),
    guardrails=[
        *STANDARD_ANALYSIS_GUARDRAILS,
        "Always begin your answer with the verdict word (Compliant / Non-Compliant / "
        "Insufficient Information) before the explanation.",
    ],
    user_template=("Policy excerpts:\n{context}\n\nSituation: {query}\n\nVerdict and reasoning:"),
    few_shot_examples=[
        FewShotExample(
            input=(
                "Policy excerpts:\n[Source: aml-policy-2025, page 1]\nEnhanced due diligence "
                "applies to domestic wire transfers over $10,000.\n\n"
                "Situation: A domestic wire transfer of $12,000 was processed without an "
                "enhanced due diligence review."
            ),
            output=(
                "Non-Compliant. The policy requires enhanced due diligence on domestic wire "
                "transfers over $10,000 [Source: aml-policy-2025, page 1]. A $12,000 transfer "
                "exceeds that threshold, so processing it without EDD review does not meet the "
                "stated policy requirement."
            ),
        )
    ],
)
