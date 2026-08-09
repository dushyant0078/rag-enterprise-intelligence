# Milestone 3 — Document Ingestion Pipeline

## What was built

**`src/parsing/`** — one parser per format, all returning the shared
`ParsedDocument` / `PageContent` / `TableContent` schema (`schemas.py`):

| Format | Parser | Notable behavior |
|---|---|---|
| PDF | `pdf_parser.py` | Native text extraction per page; pages with < 20 chars of extractable text are rasterized (PyMuPDF, 300 DPI) and OCR'd (Tesseract) automatically. Table extraction via PyMuPDF's table finder. Malformed PDFs and bad pages log a warning instead of raising. |
| DOCX | `docx_parser.py` | Walks the document body in order so paragraphs and tables interleave correctly; pseudo-paginates on hard page breaks; extracts tables. |
| PPTX | `pptx_parser.py` | One page per slide; captures shape text, tables, and speaker notes. |
| HTML | `html_parser.py` | BeautifulSoup; tables extracted structurally (not inlined as prose); strips script/style/nav/footer noise. |
| Markdown | `markdown_parser.py` | Renders to HTML (tables + fenced-code extensions) and reuses the HTML extraction logic, so headers/tables are structurally recognized. |
| CSV | `csv_parser.py` | Dialect-sniffed; kept as a real table (rows of cells), not flattened text. |
| TXT | `txt_parser.py` | Encoding fallback chain (utf-8 → utf-8-sig → latin-1) with a warning if a fallback was needed. |

`src/parsing/registry.py` maps extension → parser (`get_parser_for(path)`), the single place format support is registered.

**`src/ingestion/`** — orchestration on top of parsing:

- **Hashing** (`src/utils/hashing.py`) — SHA-256 of file bytes (identity/dedup key) and of extracted text (catches re-exports that changed bytes but not content).
- **Deduplication** (`dedup.py`) — content-hash → doc_id index; a byte-identical file ingested from a second location is marked `duplicate`, not re-embedded.
- **Version tracking** (`version_tracking.py`) — append-only per-doc_id history; an unchanged file is skipped (`skipped_unchanged`), a changed one gets a new version (`new_version`).
- **Language detection** (`language_detection.py`) — `langdetect`, seeded for reproducibility, with a safe `None` fallback on short/ambiguous text.
- **`pipeline.py`** — the `IngestionPipeline` orchestrator wiring all of the above together and writing:
  - `data/processed/<domain>/<doc_id>.json` — full parsed content
  - `data/metadata/<domain>/<doc_id>.json` — `DocumentMetadata` (hash, version, language, page count, OCR flag, parser warnings)
  - `data/metadata/ingestion_log.jsonl` — append-only audit log, one line per attempt, every status
- **`cli.py`** — `python -m src.ingestion.cli --input <dir> --domain <domain>`, with a status-count summary and a non-zero exit code on any error.

## What was actually run (not just written)

This sandbox's network allowlist doesn't reach `sec.gov` or `arxiv.org`, so real
SEC/ArXiv bulk download has to happen in an environment with open internet
access. To still prove the pipeline end-to-end here, `scripts/generate_sample_documents.py`
generates 9 small, structurally realistic documents across your target domains
(HR docx + PDF, a legal contract, an AML policy in Markdown, an insurance FAQ
in HTML, a research abstract, a technical SOP deck, a financial CSV, and —
critically — a genuinely **image-only PDF with no text layer** for the OCR path).

Ingesting all of them for real, in this sandbox, confirmed:

- **OCR fallback works**: the scanned PDF ingested with `used_ocr: true` and
  **94.9% mean Tesseract word confidence**, with the correct text recovered.
- **Table extraction works** for DOCX (`Job Level | Remote Days / Week | Approval Required`)
  and HTML (`Peril | Deductible`) — structured rows, not flattened prose.
- **Incremental ingestion works**: re-running the same domain a second time
  produced `skipped_unchanged` for every file.
- **Duplicate detection works**: copying a file to a second path and re-ingesting
  produced `duplicate`, correctly pointing `is_duplicate_of` at the original doc_id.
- **Version tracking works**: editing the legal contract's text and re-ingesting
  produced `new_version` with `version: 2`.
- **Language detection** correctly tagged every English sample `en`.

## Testing

37 pytest tests (`tests/parsing/`, `tests/ingestion/`, `tests/utils/`), 87%
coverage on new code, all passing — including a real OCR test (skipped
gracefully via `pytest.mark.skipif` if `tesseract-ocr` isn't on the machine,
so CI doesn't hard-fail on a missing system dependency, but it *does* run and
pass here). Coverage gaps are almost entirely defensive branches (encoding
fallback paths, OCR-unavailable branch, PDF error branches) that are correct
but not exercised by the current fixture set.

`make format` (black + isort), `make lint` (ruff), and `make typecheck`
(mypy) all pass clean — including fixing an isort/ruff import-sorting
disagreement by pinning `known_first_party = ["src", "api"]` in both configs,
and correcting an `[tool.mypy]` package-resolution conflict left over from
Milestone 2 (the actual import convention that emerged is `src.<subpackage>`
absolute imports everywhere, not the bare-package PYTHONPATH scheme
originally sketched — `pyproject.toml` and `src/__init__.py` now describe
what's actually there).

## Known limitations / what's explicitly deferred

- **`scripts/download_datasets.py`** (real SEC EDGAR + arXiv API fetch code)
  is deferred — it needs to be written and run somewhere with open internet
  access, since this sandbox can't reach those hosts. Say the word if you
  want it written now anyway (untestable here, but real/correct) or want to
  fold it into a later milestone once you're running this outside the
  sandbox.
- OCR language is currently a fixed default (`eng`); multi-language OCR
  routing is a natural extension once real multilingual corpora are in scope.
- PDF table extraction uses PyMuPDF's built-in finder, which is solid for
  ruled tables but not as strong as Camelot/pdfplumber on borderless tables —
  worth revisiting if financial-filing tables turn out to need it.

---
*Next: Milestone 4 — chunking and embedding system.*
