from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(
        "This is a sample English document used to exercise the ingestion pipeline end to end."
    )
    return p


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text("name,age\nAlice,30\nBob,25\n")
    return p
