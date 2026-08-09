"""Help a researcher understand or compare retrieved academic paper excerpts
— methodology, findings, comparison to other cited work."""

from __future__ import annotations

from src.prompting.guardrails import STANDARD_ANALYSIS_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

RESEARCH_ASSISTANCE_TEMPLATE = PromptTemplate(
    name="research_assistance",
    description="Explain or compare retrieved academic paper excerpts (methodology, findings, "
    "evaluation approach) faithfully to the source text.",
    system_prompt=(
        "You are a research assistant helping a scientist understand retrieved academic paper "
        "excerpts. You explain methodology, findings, and evaluation approaches faithfully to "
        "the source text, and you distinguish what a paper claims from your own assessment of "
        "it — clearly labeling any observation that goes beyond a direct restatement of the "
        "source as your own comment, not the paper's claim."
    ),
    guardrails=STANDARD_ANALYSIS_GUARDRAILS,
    user_template="Paper excerpts:\n{context}\n\nQuestion: {query}\n\nAnswer:",
    few_shot_examples=[
        FewShotExample(
            input=(
                "Paper excerpts:\n[Source: ahmos-2025-abstract]\nWe evaluate against SMOTE, "
                "ADASYN, and cost-sensitive baselines across five public benchmark datasets. Our "
                "evaluation methodology reports F1, PR-AUC, and Matthews correlation coefficient "
                "rather than accuracy alone, given severe class imbalance in every dataset "
                "considered.\n\n"
                "Question: What evaluation methodology does this paper use, and why?"
            ),
            output=(
                "The paper evaluates its method against SMOTE, ADASYN, and cost-sensitive "
                "baselines across five public benchmark datasets, reporting F1, PR-AUC, and "
                "Matthews correlation coefficient rather than accuracy alone "
                "[Source: ahmos-2025-abstract]. The stated reason is that every dataset "
                "considered has severe class imbalance, under which accuracy alone is a "
                "misleading metric [Source: ahmos-2025-abstract]."
            ),
        )
    ],
)
