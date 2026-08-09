"""Heuristic prompt injection detection at the API gateway -- a
pre-processing check that can flag/reject a request before it ever reaches
the LLM, complementing (not replacing) the INJECTION_RESISTANCE guardrail
already baked into every prompt template (Milestone 6). That guardrail
tells the model how to react if injected text reaches it; this catches
some attempts before they do.

Same honesty as every other heuristic in this project (multi-query
expansion, self-query parsing, contextual compression, lexical reranking):
a real, deterministic, testable first layer, not a claim of robust
adversarial defense. A determined attacker can phrase around a fixed
phrase list; this is a floor, not a ceiling.
"""

from __future__ import annotations

import re

_SUSPICIOUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore (?:all |the )?(?:previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (?:the |your )?system prompt", re.IGNORECASE),
    re.compile(r"you are now (?:a|an) ", re.IGNORECASE),
    re.compile(r"reveal (?:your |the )?system prompt", re.IGNORECASE),
    re.compile(r"forget (?:everything|all) (?:you were told|above)", re.IGNORECASE),
    re.compile(
        r"act as (?:if you (?:are|were)|a) (?:jailbroken|unrestricted|dan)\b", re.IGNORECASE
    ),
    re.compile(r"print (?:your |the )?(?:initial |original )?instructions", re.IGNORECASE),
)


def detect_injection_attempt(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SUSPICIOUS_PATTERNS)


def matched_patterns(text: str) -> list[str]:
    """Which pattern(s) matched, for audit logging -- not shown to the end user."""
    return [p.pattern for p in _SUSPICIOUS_PATTERNS if p.search(text)]
