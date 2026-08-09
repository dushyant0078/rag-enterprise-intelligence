from __future__ import annotations


def test_chat_grounded_answer_with_citations(client, admin_key) -> None:
    r = client.post(
        "/chat",
        json={"query": "What is the enhanced due diligence threshold for wire transfers?"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_grounded"] is True
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["chunk_id"] is not None
    assert len(body["retrieved_passages"]) >= 1
    assert "session_id" in body


def test_chat_creates_new_session_when_none_given(client, admin_key) -> None:
    r = client.post("/chat", json={"query": "test query"}, headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    assert r.json()["session_id"]


def test_chat_reuses_provided_session_id(client, admin_key) -> None:
    r1 = client.post(
        "/chat",
        json={"query": "What is the wire transfer threshold?"},
        headers={"X-API-Key": admin_key},
    )
    session_id = r1.json()["session_id"]

    r2 = client.post(
        "/chat",
        json={"query": "follow up question", "session_id": session_id},
        headers={"X-API-Key": admin_key},
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


def test_chat_off_topic_query_returns_fallback(client, admin_key) -> None:
    r = client.post(
        "/chat", json={"query": "What is the capital of France?"}, headers={"X-API-Key": admin_key}
    )
    assert r.status_code == 200
    assert r.json()["is_grounded"] is False
    assert "don't have enough information" in r.json()["answer"]


def test_query_is_stateless_no_session_returned(client, admin_key) -> None:
    r = client.post(
        "/query",
        json={"query": "What is the wire transfer threshold?"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    assert "session_id" not in r.json()


def test_query_domain_filter_narrows_results(client, admin_key) -> None:
    r = client.post(
        "/query",
        json={"query": "enhanced due diligence", "domain": "hr"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    # Filtered to hr, which has no banking content -- should not be grounded.
    assert r.json()["is_grounded"] is False


def test_query_validation_rejects_empty_query(client, admin_key) -> None:
    r = client.post("/query", json={"query": ""}, headers={"X-API-Key": admin_key})
    assert r.status_code == 422


def test_query_validation_rejects_oversized_top_k(client, admin_key) -> None:
    r = client.post(
        "/query", json={"query": "test", "top_k": 999}, headers={"X-API-Key": admin_key}
    )
    assert r.status_code == 422


def test_search_returns_raw_passages_no_generation(client, admin_key) -> None:
    r = client.post(
        "/search", json={"query": "parental leave", "top_k": 5}, headers={"X-API-Key": admin_key}
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert all("text" in item for item in body["results"])
    assert "answer" not in body


def test_search_domain_filter(client, admin_key) -> None:
    r = client.post(
        "/search",
        json={"query": "leave", "domain": "hr", "top_k": 5},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    for item in r.json()["results"]:
        assert item["domain"] == "hr"


def test_search_with_rerank_still_returns_results(client, admin_key) -> None:
    r = client.post(
        "/search",
        json={"query": "enhanced due diligence wire transfer", "rerank": True, "top_k": 5},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1


def test_query_with_compliance_checking_template(client, admin_key) -> None:
    r = client.post(
        "/query",
        json={
            "query": "A $12,000 domestic transfer was processed without review.",
            "template": "compliance_checking",
        },
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200  # doesn't error just because a different template was selected


def test_query_rejects_unknown_template(client, admin_key) -> None:
    r = client.post(
        "/query",
        json={"query": "test", "template": "not_a_real_template"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 422  # Pydantic Literal validation rejects it before the route runs


def test_chat_with_template_selection(client, admin_key) -> None:
    r = client.post(
        "/chat",
        json={"query": "What triggers enhanced due diligence?", "template": "risk_extraction"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
