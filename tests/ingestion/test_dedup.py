from __future__ import annotations

from pathlib import Path

from src.ingestion.dedup import DeduplicationIndex


def test_register_and_lookup(tmp_path: Path) -> None:
    idx = DeduplicationIndex(tmp_path / "dedup.json")
    assert idx.lookup("hash1") is None
    idx.register("hash1", "doc-a")
    assert idx.lookup("hash1") == "doc-a"
    assert len(idx) == 1


def test_first_write_wins(tmp_path: Path) -> None:
    idx = DeduplicationIndex(tmp_path / "dedup.json")
    idx.register("hash1", "doc-a")
    idx.register("hash1", "doc-b")  # should not overwrite
    assert idx.lookup("hash1") == "doc-a"


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "dedup.json"
    DeduplicationIndex(path).register("hash1", "doc-a")
    reloaded = DeduplicationIndex(path)
    assert reloaded.lookup("hash1") == "doc-a"
