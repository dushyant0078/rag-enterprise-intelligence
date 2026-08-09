from __future__ import annotations

from src.ingestion.language_detection import detect_language


def test_detects_english() -> None:
    text = "This is a fairly long sentence written in English for detection purposes."
    assert detect_language(text) == "en"


def test_short_text_returns_none() -> None:
    assert detect_language("Hi") is None


def test_empty_text_returns_none() -> None:
    assert detect_language("") is None
    assert detect_language("   ") is None
