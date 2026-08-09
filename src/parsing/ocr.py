"""OCR for scanned/image-only pages.

Uses PyMuPDF to rasterize a page to an image (no poppler/pdf2image dependency
needed) and pytesseract (Tesseract) to OCR it. Requires the ``tesseract-ocr``
system binary — see docker/README.md; this module fails with a clear,
actionable error rather than a cryptic one if it's missing.
"""

from __future__ import annotations

import shutil

import pytesseract
from PIL import Image

_TESSERACT_BINARY = "tesseract"


class OcrUnavailableError(RuntimeError):
    """Raised when the tesseract-ocr binary isn't installed on this machine."""


def tesseract_available() -> bool:
    return shutil.which(_TESSERACT_BINARY) is not None


def ocr_image(image: Image.Image, *, lang: str = "eng") -> tuple[str, float]:
    """OCR a single PIL image. Returns (text, mean_word_confidence 0-100).

    Confidence is the mean of Tesseract's per-word confidences (excluding the
    -1 sentinel it emits for non-text regions), so callers can decide whether
    an OCR pass was actually usable (see parsing/pdf_parser.py's fallback logic).
    """
    if not tesseract_available():
        raise OcrUnavailableError(
            "tesseract-ocr binary not found on PATH. Install it "
            "(e.g. `apt-get install -y tesseract-ocr`) — see docker/README.md."
        )

    text = pytesseract.image_to_string(image, lang=lang)

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    confidences = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return text, mean_confidence
