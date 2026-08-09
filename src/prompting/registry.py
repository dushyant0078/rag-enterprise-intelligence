"""Maps a use-case name to its PromptTemplate — the single place templates
are registered, mirroring the parsing/chunking/embeddings/retrieval registries.
"""

from __future__ import annotations

from src.prompting.schemas import PromptTemplate
from src.prompting.templates.compliance_checking import COMPLIANCE_CHECKING_TEMPLATE
from src.prompting.templates.contract_analysis import CONTRACT_ANALYSIS_TEMPLATE
from src.prompting.templates.financial_analysis import FINANCIAL_ANALYSIS_TEMPLATE
from src.prompting.templates.qa import QA_TEMPLATE
from src.prompting.templates.research_assistance import RESEARCH_ASSISTANCE_TEMPLATE
from src.prompting.templates.risk_extraction import RISK_EXTRACTION_TEMPLATE
from src.prompting.templates.summarization import SUMMARIZATION_TEMPLATE
from src.prompting.templates.technical_troubleshooting import TECHNICAL_TROUBLESHOOTING_TEMPLATE


class UnknownPromptTemplateError(ValueError):
    pass


_TEMPLATES: dict[str, PromptTemplate] = {
    "qa": QA_TEMPLATE,
    "summarization": SUMMARIZATION_TEMPLATE,
    "contract_analysis": CONTRACT_ANALYSIS_TEMPLATE,
    "compliance_checking": COMPLIANCE_CHECKING_TEMPLATE,
    "risk_extraction": RISK_EXTRACTION_TEMPLATE,
    "financial_analysis": FINANCIAL_ANALYSIS_TEMPLATE,
    "technical_troubleshooting": TECHNICAL_TROUBLESHOOTING_TEMPLATE,
    "research_assistance": RESEARCH_ASSISTANCE_TEMPLATE,
}


def available_templates() -> tuple[str, ...]:
    return tuple(_TEMPLATES.keys())


def get_template(name: str) -> PromptTemplate:
    template = _TEMPLATES.get(name)
    if template is None:
        raise UnknownPromptTemplateError(
            f"Unknown prompt template '{name}'. Available: {available_templates()}"
        )
    return template
