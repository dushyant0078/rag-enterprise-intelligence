from __future__ import annotations

from pathlib import Path

from src.ingestion.version_tracking import VersionTracker


def test_first_version_is_one(tmp_path: Path) -> None:
    tracker = VersionTracker(tmp_path / "versions.json")
    version = tracker.record_new_version("doc-a", "hash1", "path/a.txt")
    assert version == 1
    assert tracker.latest("doc-a").sha256 == "hash1"


def test_version_increments_on_change(tmp_path: Path) -> None:
    tracker = VersionTracker(tmp_path / "versions.json")
    tracker.record_new_version("doc-a", "hash1", "path/a.txt")
    v2 = tracker.record_new_version("doc-a", "hash2", "path/a.txt")
    assert v2 == 2
    assert len(tracker.history("doc-a")) == 2


def test_is_unchanged_true_for_same_hash(tmp_path: Path) -> None:
    tracker = VersionTracker(tmp_path / "versions.json")
    tracker.record_new_version("doc-a", "hash1", "path/a.txt")
    assert tracker.is_unchanged("doc-a", "hash1") is True
    assert tracker.is_unchanged("doc-a", "hash2") is False


def test_is_unchanged_false_for_unseen_doc(tmp_path: Path) -> None:
    tracker = VersionTracker(tmp_path / "versions.json")
    assert tracker.is_unchanged("never-seen", "hash1") is False


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "versions.json"
    VersionTracker(path).record_new_version("doc-a", "hash1", "path/a.txt")
    reloaded = VersionTracker(path)
    assert reloaded.latest("doc-a").version == 1
