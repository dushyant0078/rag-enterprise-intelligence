"""Language detection for ingested document text (langdetect)."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect's detection is non-deterministic across runs unless seeded —
# seed it once at import time so ingestion runs are reproducible.
DetectorFactory.seed = 0

_MIN_CHARS_FOR_DETECTION = 20


def detect_language(text: str) -> str | None:
    """Return an ISO 639-1 language code, or None if detection isn't reliable
    (empty/too-short text, or langdetect's own confidence failure).
    """
    if not text or len(text.strip()) < _MIN_CHARS_FOR_DETECTION:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None
