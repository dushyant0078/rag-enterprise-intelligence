"""Ingestion pipeline orchestrator.

Ties together: format detection + parsing (src/parsing/), content hashing,
deduplication, version tracking, and language detection, then writes:

- ``data/processed/<domain>/<doc_id>.json``  — the ParsedDocument (full text, pages, tables)
- ``data/metadata/<domain>/<doc_id>.json``   — the DocumentMetadata
- ``data/metadata/ingestion_log.jsonl``       — an append-only audit log, one line per attempt

This is the only module that knows about the on-disk layout under data/; every
other component (chunking, retrieval) should read the JSON files it writes,
not re-derive paths independently.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from src.ingestion.dedup import DeduplicationIndex
from src.ingestion.language_detection import detect_language
from src.ingestion.schemas import DocumentMetadata, IngestionResult, IngestionStatus
from src.ingestion.version_tracking import VersionTracker
from src.parsing.registry import UnsupportedFormatError, get_parser_for, supported_extensions
from src.utils.hashing import sha256_file, sha256_text, short_id
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "doc"


def _make_doc_id(domain: str, path: Path) -> str:
    """Stable logical identifier for a source file: readable (domain + filename
    stem) plus a short hash of the resolved parent directory, so two
    same-named files in different folders don't collide.
    """
    resolved = path.resolve()
    stem_slug = _slugify(resolved.stem)
    disambiguator = short_id(str(resolved.parent), length=8)
    return f"{_slugify(domain)}__{stem_slug}__{disambiguator}"


class IngestionPipeline:
    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        metadata_dir: str | Path = "data/metadata",
        ocr_language: str = "eng",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.dedup_index = DeduplicationIndex(self.metadata_dir / "dedup_index.json")
        self.version_tracker = VersionTracker(self.metadata_dir / "version_manifest.json")
        self.log_path = self.metadata_dir / "ingestion_log.jsonl"
        self.ocr_language = ocr_language

    # -- public API ---------------------------------------------------------

    def ingest_directory(
        self, directory: str | Path, domain: str, *, recursive: bool = True
    ) -> list[IngestionResult]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Ingestion source directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        exts = set(supported_extensions())
        files = sorted(
            f for f in directory.glob(pattern) if f.is_file() and f.suffix.lower() in exts
        )

        logger.info(
            "starting directory ingestion",
            extra={"directory": str(directory), "domain": domain, "file_count": len(files)},
        )
        return [self.ingest_file(f, domain) for f in files]

    def ingest_file(self, path: str | Path, domain: str) -> IngestionResult:
        path = Path(path)
        try:
            result = self._ingest_file_inner(path, domain)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop a batch run
            logger.exception("ingestion failed", extra={"path": str(path), "domain": domain})
            result = IngestionResult(
                doc_id=_make_doc_id(domain, path), status=IngestionStatus.ERROR, message=str(exc)
            )
        self._append_log(result)
        return result

    # -- internals ------------------------------------------------------------

    def _ingest_file_inner(self, path: Path, domain: str) -> IngestionResult:
        if not path.exists():
            raise FileNotFoundError(path)

        doc_id = _make_doc_id(domain, path)
        file_hash = sha256_file(path)

        if self.version_tracker.is_unchanged(doc_id, file_hash):
            logger.info("skipping unchanged file", extra={"doc_id": doc_id, "path": str(path)})
            return IngestionResult(doc_id=doc_id, status=IngestionStatus.SKIPPED_UNCHANGED)

        existing_owner = self.dedup_index.lookup(file_hash)
        if existing_owner is not None and existing_owner != doc_id:
            logger.info(
                "duplicate content detected",
                extra={"doc_id": doc_id, "duplicate_of": existing_owner},
            )
            metadata = DocumentMetadata(
                doc_id=doc_id,
                source_path=str(path),
                domain=domain,
                format=path.suffix.lower().lstrip("."),
                sha256=file_hash,
                content_sha256=file_hash,  # not parsed — content hash unknown, file hash stands in
                size_bytes=path.stat().st_size,
                page_count=0,
                is_duplicate_of=existing_owner,
            )
            self._write_metadata(metadata)
            return IngestionResult(
                doc_id=doc_id, status=IngestionStatus.DUPLICATE, metadata=metadata
            )

        parser = get_parser_for(path)  # raises UnsupportedFormatError for unknown extensions
        parsed = parser.parse(path)

        language = detect_language(parsed.full_text)
        is_new_doc = self.version_tracker.latest(doc_id) is None
        version = self.version_tracker.record_new_version(doc_id, file_hash, str(path))
        self.dedup_index.register(file_hash, doc_id)

        metadata = DocumentMetadata(
            doc_id=doc_id,
            source_path=str(path),
            domain=domain,
            format=parsed.format,
            sha256=file_hash,
            content_sha256=sha256_text(parsed.full_text),
            size_bytes=path.stat().st_size,
            page_count=parsed.page_count,
            language=language,
            version=version,
            used_ocr=parsed.used_ocr,
            parser_warnings=parsed.parser_warnings,
        )

        self._write_processed(domain, doc_id, parsed)
        self._write_metadata(metadata)

        status = IngestionStatus.INGESTED if is_new_doc else IngestionStatus.NEW_VERSION
        logger.info(
            "ingested document",
            extra={
                "doc_id": doc_id,
                "status": status.value,
                "version": version,
                "pages": parsed.page_count,
            },
        )
        return IngestionResult(doc_id=doc_id, status=status, metadata=metadata)

    def _write_processed(self, domain: str, doc_id: str, parsed) -> None:
        out_dir = self.processed_dir / _slugify(domain)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{doc_id}.json").write_text(parsed.model_dump_json(indent=2), encoding="utf-8")

    def _write_metadata(self, metadata: DocumentMetadata) -> None:
        out_dir = self.metadata_dir / _slugify(metadata.domain)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{metadata.doc_id}.json").write_text(
            metadata.model_dump_json(indent=2), encoding="utf-8"
        )

    def _append_log(self, result: IngestionResult) -> None:
        line = {
            "timestamp": datetime.now(UTC).isoformat(),
            "doc_id": result.doc_id,
            "status": result.status.value,
            "message": result.message,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")


__all__ = ["IngestionPipeline", "UnsupportedFormatError"]
