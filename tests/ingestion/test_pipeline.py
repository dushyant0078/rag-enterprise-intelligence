from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.schemas import IngestionStatus


@pytest.fixture()
def pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        processed_dir=tmp_path / "processed",
        metadata_dir=tmp_path / "metadata",
    )


def test_ingest_single_file(pipeline: IngestionPipeline, txt_file: Path) -> None:
    result = pipeline.ingest_file(txt_file, domain="hr")
    assert result.status == IngestionStatus.INGESTED
    assert result.metadata is not None
    assert result.metadata.domain == "hr"
    assert result.metadata.version == 1
    assert (
        result.metadata.language == "en" or result.metadata.language is None
    )  # short fixture text may be ambiguous

    # processed + metadata JSON actually landed on disk
    processed_files = list(pipeline.processed_dir.rglob("*.json"))
    metadata_files = list(pipeline.metadata_dir.rglob("*.json"))
    assert len(processed_files) == 1
    assert any(
        f.name.endswith(".json") and "dedup" not in f.name and "version_manifest" not in f.name
        for f in metadata_files
    )


def test_reingesting_unchanged_file_is_skipped(pipeline: IngestionPipeline, txt_file: Path) -> None:
    first = pipeline.ingest_file(txt_file, domain="hr")
    second = pipeline.ingest_file(txt_file, domain="hr")
    assert first.status == IngestionStatus.INGESTED
    assert second.status == IngestionStatus.SKIPPED_UNCHANGED
    assert second.doc_id == first.doc_id


def test_editing_file_creates_new_version(pipeline: IngestionPipeline, txt_file: Path) -> None:
    first = pipeline.ingest_file(txt_file, domain="hr")
    txt_file.write_text(txt_file.read_text() + "\nAn additional line changing the content.")
    second = pipeline.ingest_file(txt_file, domain="hr")

    assert first.status == IngestionStatus.INGESTED
    assert second.status == IngestionStatus.NEW_VERSION
    assert second.doc_id == first.doc_id
    assert second.metadata.version == 2


def test_duplicate_content_detected_across_paths(
    pipeline: IngestionPipeline, tmp_path: Path
) -> None:
    original = tmp_path / "original.txt"
    original.write_text("identical content shared by two files")
    copy = tmp_path / "subdir" / "copy.txt"
    copy.parent.mkdir()
    copy.write_text("identical content shared by two files")

    first = pipeline.ingest_file(original, domain="hr")
    second = pipeline.ingest_file(copy, domain="hr")

    assert first.status == IngestionStatus.INGESTED
    assert second.status == IngestionStatus.DUPLICATE
    assert second.metadata.is_duplicate_of == first.doc_id


def test_ingest_directory_processes_all_supported_files(
    pipeline: IngestionPipeline, tmp_path: Path, txt_file: Path, csv_file: Path
) -> None:
    directory = txt_file.parent
    # txt_file and csv_file fixtures already share tmp_path in this test's context via pytest's
    # per-test tmp_path — but to be explicit, ingest the directory both files live in.
    results = pipeline.ingest_directory(directory, domain="mixed")
    statuses = {r.status for r in results}
    assert IngestionStatus.INGESTED in statuses
    assert len(results) >= 2


def test_ingest_directory_missing_raises(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_directory(tmp_path / "does_not_exist", domain="hr")


def test_ingest_file_missing_returns_error_result_not_raise(
    pipeline: IngestionPipeline, tmp_path: Path
) -> None:
    result = pipeline.ingest_file(tmp_path / "missing.txt", domain="hr")
    assert result.status == IngestionStatus.ERROR
    assert result.message is not None


def test_ingestion_log_is_appended(pipeline: IngestionPipeline, txt_file: Path) -> None:
    pipeline.ingest_file(txt_file, domain="hr")
    assert pipeline.log_path.exists()
    lines = pipeline.log_path.read_text().strip().splitlines()
    assert len(lines) == 1
