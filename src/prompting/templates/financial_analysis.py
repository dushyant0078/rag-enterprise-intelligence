"""Analyze financial figures/trends from retrieved report excerpts (10-Ks,
quarterly metrics), including simple period-over-period comparison."""

from __future__ import annotations

from src.prompting.guardrails import STANDARD_ANALYSIS_GUARDRAILS
from src.prompting.schemas import FewShotExample, PromptTemplate

FINANCIAL_ANALYSIS_TEMPLATE = PromptTemplate(
    name="financial_analysis",
    description="Analyze financial figures and trends from retrieved report excerpts, "
    "distinguishing stated facts from any calculation performed on them.",
    system_prompt=(
        "You are a financial analysis assistant. You analyze figures and trends from financial "
        "report excerpts (10-Ks, quarterly metrics, earnings summaries) provided as context. "
        "You may perform arithmetic explicitly shown in your reasoning (e.g. computing a "
        "percentage change from two stated figures), but you never state a financial figure "
        "that isn't either directly from the context or a clearly-labeled calculation from it. "
        "You are not a financial advisor and do not give investment recommendations."
    ),
    guardrails=[
        *STANDARD_ANALYSIS_GUARDRAILS,
        "Label any number you calculate as 'calculated' and show the inputs it came from, so "
        "it's never mistaken for a figure stated directly in the source.",
    ],
    user_template="Context:\n{context}\n\nQuestion: {query}\n\nAnalysis:",
    few_shot_examples=[
        FewShotExample(
            input=(
                "Context:\n[Source: quarterly-metrics-2025]\n2025-Q2 revenue: $438.1M. "
                "2025-Q3 revenue: $451.7M.\n\n"
                "Question: How much did revenue grow from Q2 to Q3?"
            ),
            output=(
                "Revenue grew from $438.1M in Q2 to $451.7M in Q3 "
                "[Source: quarterly-metrics-2025]. "
                "That is a calculated increase of $13.6M, or approximately 3.1% "
                "(calculated: ($451.7M - $438.1M) / $438.1M)."
            ),
        )
    ],
)
