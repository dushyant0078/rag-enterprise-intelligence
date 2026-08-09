from __future__ import annotations

import io


def test_upload_and_reindex_makes_document_searchable(client, admin_key) -> None:
    file_content = b"Employees may expense up to $500 per trip for business travel."
    r = client.post(
        "/upload",
        files={"file": ("travel_policy.txt", io.BytesIO(file_content), "text/plain")},
        data={"domain": "hr"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ingested"
    doc_id = r.json()["doc_id"]

    r = client.post("/reindex", json={}, headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 3  # the 2 seeded docs + the new one

    r = client.post(
        "/query",
        json={"query": "How much can I expense per trip?"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    assert r.json()["is_grounded"] is True
    assert any(c["doc_id"] == doc_id for c in r.json()["citations"])


def test_upload_rejects_unsupported_file_type(client, admin_key) -> None:
    r = client.post(
        "/upload",
        files={"file": ("f.exe", io.BytesIO(b"binary junk"), "application/octet-stream")},
        data={"domain": "hr"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 415


def test_upload_requires_auth(client) -> None:
    r = client.post(
        "/upload",
        files={"file": ("f.txt", io.BytesIO(b"content"), "text/plain")},
        data={"domain": "hr"},
    )
    assert r.status_code == 401


def test_reindex_before_any_documents_exist_still_works_with_seeded_corpus(
    client, admin_key
) -> None:
    # The isolated_data_dirs fixture always seeds 2 documents, so reindex succeeds even with
    # nothing newly uploaded.
    r = client.post("/reindex", json={}, headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1


def test_reupload_same_file_is_idempotent(client, admin_key) -> None:
    file_content = b"Employees may expense up to $500 per trip for business travel."
    r1 = client.post(
        "/upload",
        files={"file": ("travel_policy.txt", io.BytesIO(file_content), "text/plain")},
        data={"domain": "hr"},
        headers={"X-API-Key": admin_key},
    )
    r2 = client.post(
        "/upload",
        files={"file": ("travel_policy.txt", io.BytesIO(file_content), "text/plain")},
        data={"domain": "hr"},
        headers={"X-API-Key": admin_key},
    )
    assert r1.json()["status"] == "ingested"
    assert r2.json()["status"] == "skipped_unchanged"
