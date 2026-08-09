from __future__ import annotations


def test_submit_feedback_after_chat(client, admin_key) -> None:
    chat_response = client.post(
        "/chat", json={"query": "What is the enhanced due diligence threshold?"},
        headers={"X-API-Key": admin_key},
    )  # fmt: skip
    session_id = chat_response.json()["session_id"]

    r = client.post(
        "/feedback",
        json={
            "session_id": session_id,
            "rating": "up",
            "query": "test",
            "answer_excerpt": "answer",
        },
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 200
    assert r.json()["recorded"] is True


def test_submit_feedback_rejects_invalid_rating(client, admin_key) -> None:
    r = client.post(
        "/feedback",
        json={"session_id": "s1", "rating": "sideways"},
        headers={"X-API-Key": admin_key},
    )
    assert r.status_code == 422  # Pydantic Literal validation


def test_submit_feedback_requires_auth(client) -> None:
    r = client.post("/feedback", json={"session_id": "s1", "rating": "up"})
    assert r.status_code == 401


def test_monitoring_summary_reflects_real_chat_traffic(client, admin_key) -> None:
    client.post(
        "/chat", json={"query": "What is the enhanced due diligence threshold?"},
        headers={"X-API-Key": admin_key},
    )  # fmt: skip
    client.post(
        "/query",
        json={"query": "How many weeks of parental leave?"},
        headers={"X-API-Key": admin_key},
    )

    r = client.get("/monitoring/summary", headers={"X-API-Key": admin_key})
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval"]["observations"] >= 2
    assert "estimated_cost_usd" in body
    assert "hallucination" in body


def test_monitoring_summary_requires_auth(client) -> None:
    r = client.get("/monitoring/summary")
    assert r.status_code == 401


def test_monitoring_summary_reflects_feedback(client, admin_key) -> None:
    chat_response = client.post("/chat", json={"query": "test"}, headers={"X-API-Key": admin_key})
    session_id = chat_response.json()["session_id"]
    client.post(
        "/feedback",
        json={"session_id": session_id, "rating": "down"},
        headers={"X-API-Key": admin_key},
    )

    r = client.get("/monitoring/summary", headers={"X-API-Key": admin_key})
    assert r.json()["feedback"]["total"] == 1
    assert r.json()["feedback"]["down"] == 1


def test_feedback_isolated_from_other_test_runs(client, admin_key) -> None:
    """Regression test: the isolated_data_dirs fixture originally set every
    RAG_*_DIR env var except RAG_FEEDBACK_DIR, so FeedbackStore silently
    fell back to the real project's data/feedback/ directory -- every test
    run polluted the same real folder, and a later test's feedback count
    included earlier tests' records. Caught by an assertion on an exact
    count, not by "at least one record exists". A single feedback
    submission in a fresh client must produce exactly one record, not more.
    """
    chat_response = client.post(
        "/chat", json={"query": "isolated test"}, headers={"X-API-Key": admin_key}
    )
    client.post(
        "/feedback",
        json={"session_id": chat_response.json()["session_id"], "rating": "up"},
        headers={"X-API-Key": admin_key},
    )
    r = client.get("/monitoring/summary", headers={"X-API-Key": admin_key})
    assert r.json()["feedback"]["total"] == 1
