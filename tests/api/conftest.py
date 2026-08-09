from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.parsing.schemas import PageContent, ParsedDocument


def _write_doc(processed_dir: Path, domain: str, doc_id: str, text: str) -> None:
    """doc_id must follow the real ingestion pipeline's "domain__slug__hash"
    naming convention (src/ingestion/pipeline.py's _make_doc_id) -- domain
    filtering (src/memory/pipeline_factory.py's load_processed_documents)
    derives the domain from that prefix, not from which folder the file
    happens to sit in. Using a plain id like "hr-doc" here initially broke
    every RBAC/domain-filter test silently (retrieval found zero results,
    not the wrong ones) -- caught by an end-to-end API test actually
    asserting on grounding, not by the folder-placement alone looking right.
    """
    full_doc_id = f"{domain}__{doc_id}"
    doc = ParsedDocument(
        source_path=f"{full_doc_id}.txt",
        format="txt",
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        page_count=1,
    )
    out_dir = processed_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{full_doc_id}.json").write_text(doc.model_dump_json(), encoding="utf-8")


@pytest.fixture()
def isolated_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the app at tmp directories via env vars (see api/main.py's
    lifespan) instead of the real project's data/ -- every API test runs
    against a small, seeded, disposable corpus, never the real one.
    """
    processed_dir = tmp_path / "processed"
    monkeypatch.setenv("RAG_PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_RAW_UPLOAD_DIR", str(tmp_path / "raw_uploads"))
    monkeypatch.setenv("RAG_METADATA_DIR", str(tmp_path / "metadata"))
    monkeypatch.setenv("RAG_FEEDBACK_DIR", str(tmp_path / "feedback"))

    _write_doc(
        processed_dir,
        "banking",
        "banking-doc",
        "Wire transfers over $10,000 require enhanced due diligence review by compliance.",
    )
    _write_doc(
        processed_dir,
        "hr",
        "hr-doc",
        "Employees in their first year get twelve weeks of paid parental leave.",
    )
    return processed_dir


@pytest.fixture()
def client(isolated_data_dirs: Path):
    # Import main *after* env vars are set (isolated_data_dirs fixture already ran) so the
    # lifespan below reads the tmp paths; the app module itself is import-cached, but the
    # lifespan function reads os.environ at call time (on `with TestClient(...)` entry), not
    # at import time, so this ordering is safe even on repeated test-session imports.
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def admin_key(client: TestClient) -> str:
    registry = client.app.state.app_state.api_key_registry
    return next(k for k, v in registry._keys.items() if v.allowed_domains is None)


@pytest.fixture()
def hr_key(client: TestClient) -> str:
    registry = client.app.state.app_state.api_key_registry
    return next(k for k, v in registry._keys.items() if v.allowed_domains == ["hr"])
