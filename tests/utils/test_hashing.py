from __future__ import annotations

from pathlib import Path

from src.utils.hashing import sha256_file, sha256_text, short_id


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    assert sha256_file(f) == sha256_file(f)


def test_sha256_file_differs_for_different_content(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello world")
    f2.write_text("goodbye world")
    assert sha256_file(f1) != sha256_file(f2)


def test_sha256_file_matches_content_regardless_of_filename(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("identical content")
    f2.write_text("identical content")
    assert sha256_file(f1) == sha256_file(f2)


def test_sha256_text_deterministic() -> None:
    assert sha256_text("some text") == sha256_text("some text")
    assert sha256_text("some text") != sha256_text("other text")


def test_short_id_is_stable_and_short() -> None:
    a = short_id("some/path")
    b = short_id("some/path")
    c = short_id("some/other/path")
    assert a == b
    assert a != c
    assert len(a) == 12
    assert len(short_id("x", length=6)) == 6
