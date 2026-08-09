from __future__ import annotations


def test_get_document_returns_full_content(client, admin_key) -> None:
    r = client.get("/documents/hr__hr-doc", headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "hr__hr-doc"
    assert body["domain"] == "hr"
    assert "parental leave" in body["full_text"].lower()


def test_get_document_unknown_id_returns_404(client, admin_key) -> None:
    r = client.get("/documents/does-not-exist", headers={"X-API-Key": admin_key})
    assert r.status_code == 404


def test_get_document_requires_auth(client) -> None:
    r = client.get("/documents/hr__hr-doc")
    assert r.status_code == 401


def test_get_document_rbac_blocks_out_of_scope_domain(client, hr_key) -> None:
    r = client.get("/documents/banking__banking-doc", headers={"X-API-Key": hr_key})
    assert r.status_code == 403


def test_get_document_rbac_allows_in_scope_domain(client, hr_key) -> None:
    r = client.get("/documents/hr__hr-doc", headers={"X-API-Key": hr_key})
    assert r.status_code == 200


def test_get_session_history_after_chat(client, admin_key) -> None:
    r1 = client.post("/chat", json={"query": "test question"}, headers={"X-API-Key": admin_key})
    session_id = r1.json()["session_id"]

    r2 = client.get(f"/sessions/{session_id}", headers={"X-API-Key": admin_key})
    assert r2.status_code == 200
    body = r2.json()
    assert body["session_id"] == session_id
    assert len(body["turns"]) == 2
    assert body["turns"][0]["role"] == "user"
    assert body["turns"][1]["role"] == "assistant"


def test_get_session_history_unknown_session_returns_404(client, admin_key) -> None:
    r = client.get("/sessions/does-not-exist", headers={"X-API-Key": admin_key})
    assert r.status_code == 404


def test_get_session_history_requires_auth(client) -> None:
    r = client.get("/sessions/some-id")
    assert r.status_code == 401


def test_scoped_key_cannot_view_another_users_session(client, admin_key, hr_key) -> None:
    r1 = client.post("/chat", json={"query": "admin question"}, headers={"X-API-Key": admin_key})
    session_id = r1.json()["session_id"]

    r2 = client.get(f"/sessions/{session_id}", headers={"X-API-Key": hr_key})
    assert r2.status_code == 403


def test_scoped_key_can_view_own_session(client, hr_key) -> None:
    r1 = client.post("/chat", json={"query": "hr question"}, headers={"X-API-Key": hr_key})
    session_id = r1.json()["session_id"]

    r2 = client.get(f"/sessions/{session_id}", headers={"X-API-Key": hr_key})
    assert r2.status_code == 200
