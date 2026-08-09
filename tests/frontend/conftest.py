"""A real, live FastAPI server running in a background thread for the
frontend test suite. Streamlit's AppTest exercises the actual page scripts,
which call frontend.api_client.RAGAPIClient over real HTTP -- pointing that
client at an in-process TestClient wouldn't exercise the same code path a
user's browser session does, so this starts an actual uvicorn server on a
free local port instead, the same rigor as every other milestone's
end-to-end verification in this project.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn
from streamlit.testing.v1 import AppTest

from src.parsing.schemas import PageContent, ParsedDocument

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def _write_doc(processed_dir: Path, domain: str, doc_id: str, text: str) -> None:
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_api_server(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("frontend_live_api")
    processed_dir = tmp_path / "processed"

    _write_doc(
        processed_dir,
        "banking",
        "aml-doc",
        "Wire transfers over $10,000 require enhanced due diligence review by compliance. "
        "International wire transfers over $3,000 also require enhanced due diligence.",
    )
    _write_doc(
        processed_dir,
        "hr",
        "leave-doc",
        "Employees in their first year get twelve weeks of paid parental leave.",
    )
    _write_doc(
        processed_dir,
        "legal",
        "msa-doc",
        "Either party may terminate this agreement for convenience upon sixty days written notice.",
    )

    os.environ["RAG_PROCESSED_DIR"] = str(processed_dir)
    os.environ["RAG_INDEX_DIR"] = str(tmp_path / "index")
    os.environ["RAG_RAW_UPLOAD_DIR"] = str(tmp_path / "raw_uploads")
    os.environ["RAG_METADATA_DIR"] = str(tmp_path / "metadata")
    os.environ["RAG_FEEDBACK_DIR"] = str(tmp_path / "feedback")

    from api.main import app

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):
        try:
            requests.get(f"{base_url}/health", timeout=1)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        raise RuntimeError("live_api_server did not become ready in time")

    yield base_url, app.state.app_state.api_key_registry

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def admin_key(live_api_server) -> str:
    _, registry = live_api_server
    return next(k for k, v in registry._keys.items() if v.allowed_domains is None)


@pytest.fixture()
def hr_key(live_api_server) -> str:
    _, registry = live_api_server
    return next(k for k, v in registry._keys.items() if v.allowed_domains == ["hr"])


@pytest.fixture()
def app_test(live_api_server, admin_key):
    """Returns a factory: app_test("pages/1_chat_assistant.py") -> a fresh
    AppTest instance for that script, pre-populated with the live server's
    base_url and an admin API key already in session_state -- every page
    test starts from "connected," not from re-deriving connection state.
    """
    base_url, _ = live_api_server

    def _build(relative_path: str) -> AppTest:
        at = AppTest.from_file(str(FRONTEND_DIR / relative_path))
        at.session_state["api_base_url"] = base_url
        at.session_state["api_key"] = admin_key
        return at

    return _build
